"""The horde field (gameplay pack v1 G9p; the residency design's P3d).

``docs/residency_batching_design_2026-11-07.md`` §4.9 and
``docs/bots_zombies_speeds_design_2026-11-10.md`` §3 (Erik's ruling of
2026-11-10: the distance field moves forward, and the horde is rewritten
once, onto it).

THE RULE: the horde paths ONLY by an integer distance field; a zombie
behaviour is a MASK, a GOAL SET or a RULE ON THE DESCENT, never a per-zombie
search. A field is a breadth-first flood over the ANCHOR grid of a footprint
``fp`` (cell ``(y, x)`` is the top-left tile of an ``fp x fp`` body; it is
passable when that whole block is in bounds and every tile of it is
``tile_ok``), from a set of seed cells, 4-connected; a cell the flood never
reaches holds :data:`UNREACHED`. A zombie DESCENDS a field: of its four
neighbours it steps to the lowest one strictly below its own cell's value
that it can enter now, ties in the cell's fixed order
(:data:`DESCENT_ORDER`) -- or it stays. Its every step is to a strictly
lower value.

- THE PREY FIELD (a goal set): seeded at every living non-zombie (each
  prey's centre tile in the zombie's anchor space, ``centre - fp // 2``: the
  anchor that would centre the zombie on the prey, which is A*'s old goal
  when the footprints are equal), so a zombie walks to the PATH-nearest prey.
- A NOISE FIELD (a goal set, stage 2): seeded at one heard noise's tile, the
  same way; at most ``[zombie] noise_fields_max`` distinct noise targets are
  live (the newest), so the fields stay few.
- FEAR OF FIRE (a mask, stage 2): a burning tile or one whose temperature
  mirror is at or above ``[zombie] fear_temp`` (game degrees above ambient,
  an integer compare) is not ``tile_ok`` at the refresh, so the horde routes
  around fire, and the descent never steps a block onto one between
  refreshes (:func:`block_feared`). A zombie the feared field does not reach
  (fire closes every way) walks the same goal set flooded WITHOUT the fear
  (``fearless=True``) -- it comes up to the fire and waits at its edge,
  since its steps still refuse every feared block.
- DOOR BASHING (a mask and a rule, stage 2): a closed door that is not a
  hull door (no span tile borders vacuum, the ambient ring or the map's
  edge) is ``tile_ok``; a descent whose step is blocked only by such a door
  STRIKES it instead (:func:`descend` returns ``"bash"``).
- The flood itself is C++ (``PhysicsEngine.horde_field``, ``cpp/src/
  horde_field.cpp``) whenever the map has an engine bound -- the played game
  and every match; :func:`flood_reference` is the Python reference of the
  same law for a bare ``GameMap`` (tests, tools), the ``stamp_units``
  precedent. The device kernel is residency P7d.
- REFRESH (:class:`HordeState`): every field (and the mask) is dropped at the
  zombie slot of a tick ``t`` with ``t % field_refresh_ticks == 0`` and
  whenever the map's topology serial (``GameMap._tile_serial``: a wall
  destroyed, a door opened, closed or broken) moved since the last slot; a
  field is flooded the first time a zombie needs it after a drop, from the
  state of THAT tick. So the field a zombie reads at tick ``t`` was flooded
  at the first tick since the last drop on which one was needed -- a pure
  function of the synced history. The cache is never synced and never
  digested (a derived quantity, the vision cache's precedent). ACCEPTED GAP:
  a save / load between two refreshes would re-flood from the loaded tick's
  state.
- THE WANDER (stage 2) draws no random number: a dormant zombie's step tick
  and direction are a pure function of (its id, the tick) -- the vendored
  Philox block used as a seedless HASH (key 0, counter ``(tick, 0,``
  :data:`WANDER_SALT` ``, id)``), never a stream.

Integer throughout; nothing here reads a float field.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from simulation.rng_salts import Salt as _Salt   # #29 P3b: the one salt enum

#: The value of a cell the flood never reached (no path: the zombie waits).
#: Equal to ``horde::UNREACHED`` in ``cpp/src/horde_field.h``.
UNREACHED = 0x7FFFFFFF

#: The four neighbours, ``(dy, dx)``: north, west, east, south (the
#: row-major order of the four cells) -- the flood's order (irrelevant to
#: its distances) and the wander's direction index.
NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))

#: The descent's TIE order, by the parity of the cell's ``x + y``: vertical
#: first (north, south, west, east) on an even cell, horizontal first (west,
#: east, north, south) on an odd one. On a 4-connected field every cell
#: toward a diagonal goal has two equal lower neighbours; one fixed order
#: would walk an L (all north, then all east), this one walks the staircase
#: (north, east, north, ...), which is the straight line the 4-grid allows.
#: A fixed rule per cell, so the device port is the same two tables.
DESCENT_ORDER = (((-1, 0), (1, 0), (0, -1), (0, 1)),
                 ((0, -1), (0, 1), (-1, 0), (1, 0)))

#: The wander hash's counter salt (ASCII "WAND"; the one salt enum's,
#: ``rng_salts.Salt.WANDER``, #29 P3b) and the per-id stagger of the
#: wander's step ticks (odd, so ids spread over the period).
WANDER_SALT = int(_Salt.WANDER)
WANDER_STAGGER = 37
_M32 = 0xFFFFFFFF

#: The four behaviours' switches, in ``[zombie]`` (each a bool): all four
#: off (and the sprint) is the stage-1 field law exactly.
BEHAVIOUR_SWITCHES = ("wander", "hearing", "fear_fire", "bash_doors",
                      "sprint")


# ---------------------------------------------------------------------------
# The settings door
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HordeSettings:
    """``[zombie]``'s horde dials, quantized ONCE for one map
    (:func:`horde_settings`). Radii are integer tiles, compared as squares
    (the law's old ``math.sqrt`` gates, integer now); durations are ticks."""
    trigger_r2: int
    propagation_r2: int
    field_refresh_ticks: int
    wander: bool = False
    wander_period_ticks: int = 1
    wander_leash_r2: int = 0
    hearing: bool = False
    noise_timeout_ticks: int = 1
    noise_fields_max: int = 1
    noise_arrive_tiles: int = 0
    fear_fire: bool = False
    fear_temp_q16: int = 0
    bash_doors: bool = False
    door_bash_damage: float = 0.0
    bash_loudness_tiles: int = 0
    sprint: bool = False
    sprint_r2: int = 0


def _key(section, key):
    if section is None or not hasattr(section, key):
        raise ValueError(f"[zombie] is missing {key!r}")
    return getattr(section, key)


def _int_key(section, key, lo=None):
    v = _key(section, key)
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise ValueError(f"[zombie] {key} must be an integer, got {v!r}")
    if lo is not None and v < lo:
        raise ValueError(f"[zombie] {key} must be >= {lo}, got {v!r}")
    return int(v)


def _bool_key(section, key):
    v = _key(section, key)
    if not isinstance(v, bool):
        raise ValueError(f"[zombie] {key} must be true or false, got {v!r}")
    return v


def _num_key(section, key, *, positive=True):
    v = _key(section, key)
    if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v):
        raise ValueError(f"[zombie] {key} must be a finite number, got {v!r}")
    if positive and not v > 0:
        raise ValueError(f"[zombie] {key} must be > 0, got {v!r}")
    if not positive and v < 0:
        raise ValueError(f"[zombie] {key} must be >= 0, got {v!r}")
    return float(v)


def _tiles(metres: float, tile: float) -> int:
    """Metres as whole tiles, at least one (the weapon range's rule)."""
    t = int(metres / tile + 0.5)
    return t if t > 1 else 1


def horde_settings(cfg, tile_size_m: float = 1.0) -> HordeSettings:
    """``[zombie]`` of a loaded config through the refusing door: every key
    present and of its type (ValueError naming it otherwise), quantized
    ONCE for a map of ``tile_size_m`` metres per tile at ``[clock]
    ticks_per_second``."""
    from config import ticks_from_seconds
    from simulation.gas_fixed import quantize_q16_scalar
    from simulation.noise import radius_tiles
    z = getattr(cfg, "zombie", None)
    if z is None:
        raise ValueError("config has no [zombie] table")
    tile = float(tile_size_m)
    if not tile > 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m!r}")
    tps = int(cfg.clock.ticks_per_second)
    trig = _int_key(z, "trigger_radius", 0)
    prop = _int_key(z, "propagation_radius", 0)
    leash = _tiles(_num_key(z, "wander_leash_m"), tile)
    sprint_r = _tiles(_num_key(z, "sprint_radius_m"), tile)
    fear = _num_key(z, "fear_temp")
    fear_q = quantize_q16_scalar(fear)
    if fear_q <= 0:
        raise ValueError(f"[zombie] fear_temp {fear!r} quantizes to 0")
    return HordeSettings(
        trigger_r2=trig * trig,
        propagation_r2=prop * prop,
        field_refresh_ticks=_int_key(z, "field_refresh_ticks", 1),
        wander=_bool_key(z, "wander"),
        wander_period_ticks=ticks_from_seconds(
            _num_key(z, "wander_period_seconds"), tps),
        wander_leash_r2=leash * leash,
        hearing=_bool_key(z, "hearing"),
        noise_timeout_ticks=ticks_from_seconds(
            _num_key(z, "noise_timeout_seconds"), tps),
        noise_fields_max=_int_key(z, "noise_fields_max", 1),
        noise_arrive_tiles=radius_tiles(
            _num_key(z, "noise_arrive_m", positive=False), tile),
        fear_fire=_bool_key(z, "fear_fire"),
        fear_temp_q16=fear_q,
        bash_doors=_bool_key(z, "bash_doors"),
        door_bash_damage=_num_key(z, "door_bash_damage"),
        bash_loudness_tiles=radius_tiles(
            _num_key(z, "bash_loudness_m", positive=False), tile),
        sprint=_bool_key(z, "sprint"),
        sprint_r2=sprint_r * sprint_r,
    )


# ---------------------------------------------------------------------------
# The flood
# ---------------------------------------------------------------------------
def block_passable(tile_ok: np.ndarray, fp: int) -> np.ndarray:
    """The anchor grid's passability: ``True`` at ``(y, x)`` when the ``fp x
    fp`` block from there is in bounds and all ``tile_ok`` (an integral
    image of the bad tiles; integer)."""
    h, w = tile_ok.shape
    bad = np.zeros((h + 1, w + 1), dtype=np.int64)
    bad[1:, 1:] = np.cumsum(np.cumsum(~tile_ok.astype(bool), axis=0), axis=1)
    out = np.zeros((h, w), dtype=bool)
    if fp <= h and fp <= w:
        s = (bad[fp:, fp:] - bad[:h + 1 - fp, fp:]
             - bad[fp:, :w + 1 - fp] + bad[:h + 1 - fp, :w + 1 - fp])
        out[:h + 1 - fp, :w + 1 - fp] = s == 0
    return out


def flood_reference(tile_ok: np.ndarray, fp: int, seeds) -> np.ndarray:
    """The flood's Python reference (a bare map without an engine): the
    law of ``horde::flood`` -- seeds labelled 0 whether passable or not,
    the frontier entering passable anchor cells only, 4-connected; every
    unreached cell :data:`UNREACHED`."""
    h, w = tile_ok.shape
    passable = block_passable(tile_ok, fp)
    dist = np.full((h, w), UNREACHED, dtype=np.int32)
    q = deque()
    for y, x in seeds:
        y, x = int(y), int(x)
        if 0 <= y < h and 0 <= x < w and dist[y, x] != 0:
            dist[y, x] = 0
            q.append((y, x))
    while q:
        y, x = q.popleft()
        nd = int(dist[y, x]) + 1
        for dy, dx in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and passable[ny, nx]
                    and dist[ny, nx] == UNREACHED):
                dist[ny, nx] = nd
                q.append((ny, nx))
    return dist


def flood(gmap, tile_ok: np.ndarray, fp: int, seeds) -> np.ndarray:
    """THE flood: ``PhysicsEngine.horde_field`` when ``gmap`` has an engine
    bound, else :func:`flood_reference` (the same law)."""
    seeds_arr = np.ascontiguousarray(
        np.asarray(seeds, dtype=np.int32).reshape(-1, 2))
    eng = getattr(gmap, "_physics_engine", None)
    if eng is None:
        return flood_reference(tile_ok, fp, seeds_arr)
    out = np.empty(tile_ok.shape, dtype=np.int32)
    eng.horde_field(np.ascontiguousarray(tile_ok, dtype=np.uint8), int(fp),
                    seeds_arr, out)
    return out


def walkable_tiles(gmap) -> np.ndarray:
    """The tiles a body may overlap: the mobility table's ``mobility > 0``
    projected through the material grid -- ``GameMap.is_passable_block``'s
    own predicate, tile by tile."""
    return gmap.materials.mobility[gmap.material] > 0


def feared_tiles(gmap, fear_temp_q16: int) -> np.ndarray:
    """The tiles a zombie fears: burning (``fire > 0``) or at or above
    ``fear_temp_q16`` on the temperature mirror (Q16 game degrees above
    ambient; an integer compare)."""
    return (gmap.fire > 0) | (gmap.temperature >= int(fear_temp_q16))


def block_feared(gmap, y: int, x: int, fp: int, fear_temp_q16: int) -> bool:
    """Whether the ``fp`` block at anchor ``(y, x)`` holds a feared tile
    NOW (the live map, between refreshes)."""
    ys = slice(max(0, y), max(0, y + fp))
    xs = slice(max(0, x), max(0, x + fp))
    return bool((gmap.fire[ys, xs] > 0).any()
                or (gmap.temperature[ys, xs] >= int(fear_temp_q16)).any())


def is_hull_door(gmap, span) -> bool:
    """A door a zombie never bashes: one of its span tiles borders (4-way,
    outside the span) vacuum, the ambient ring or the map's edge -- the far
    side of a hull door is space, and a zombie does not open the ship."""
    h, w = gmap.material.shape
    spanset = set((int(a), int(b)) for a, b in span)
    vac = gmap.is_vacuum
    amb = getattr(gmap, "is_ambient", None)
    for fy, fx in spanset:
        for dy, dx in NEIGHBOURS:
            ny, nx = fy + dy, fx + dx
            if (ny, nx) in spanset:
                continue
            if not (0 <= ny < h and 0 <= nx < w):
                return True
            if vac[ny, nx] or (amb is not None and amb[ny, nx]):
                return True
    return False


def bashable_door_tiles(gmap, doors) -> np.ndarray:
    """The closed-door tiles the horde may bash: every ``MAT_DOOR_CLOSED``
    span tile of a living, CLOSED door entity that is not a hull door."""
    from simulation.entities.door import DOOR_CLOSED
    from simulation.materials import MAT_DOOR_CLOSED
    mask = np.zeros(gmap.material.shape, dtype=bool)
    for d in doors or ():
        if not getattr(d, "alive", True) or d.state != DOOR_CLOSED:
            continue
        if is_hull_door(gmap, d.span):
            continue
        for fy, fx in d.span:
            if int(gmap.material[fy, fx]) == MAT_DOOR_CLOSED:
                mask[fy, fx] = True
    return mask


def anchor_of(cy: int, cx: int, fp: int) -> tuple:
    """The anchor cell that centres an ``fp`` body on tile ``(cy, cx)``."""
    return (int(cy) - fp // 2, int(cx) - fp // 2)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
class HordeState:
    """One map's field cache (never synced, never digested): the fields
    flooded since the last drop, keyed ``("prey", fp)`` / ``("noise", fp,
    y, x)``; the tile mask they were flooded over and the bashable door
    tiles of that mask. A Simulation keeps one per reset; a direct caller of
    the law with none gets a fresh one per call."""

    def __init__(self):
        self.fields: dict = {}
        self.tile_ok = None
        self.tile_ok_fearless = None
        self.bashable = None
        self.serial = None
        self.settings = None
        self.doors = ()
        self.floods = 0              # diagnostics: floods computed

    def begin_tick(self, gmap, tick: int, settings: HordeSettings,
                   doors=()) -> None:
        """The slot head: drop every field at a refresh tick or when the
        topology moved since the last slot."""
        self.settings = settings
        self.doors = doors or ()
        serial = getattr(gmap, "_tile_serial", 0)
        if (int(tick) % settings.field_refresh_ticks == 0
                or serial != self.serial):
            self.fields.clear()
            self.tile_ok = None
            self.tile_ok_fearless = None
            self.bashable = None
            self.serial = serial

    def mask(self, gmap, fearless: bool = False) -> np.ndarray:
        """The tile mask of this refresh (built once per drop): the walkable
        tiles, plus the bashable doors (``bash_doors``), minus the feared
        tiles (``fear_fire``) unless ``fearless``."""
        if self.tile_ok_fearless is None:
            ok = walkable_tiles(gmap)
            s = self.settings
            if s is not None and s.bash_doors:
                self.bashable = bashable_door_tiles(gmap, self.doors)
                ok = ok | self.bashable
            self.tile_ok_fearless = ok
            if s is not None and s.fear_fire:
                ok = ok & ~feared_tiles(gmap, s.fear_temp_q16)
            self.tile_ok = ok
        return self.tile_ok_fearless if fearless else self.tile_ok

    def _field(self, gmap, key, fp, seeds, fearless=False) -> np.ndarray:
        f = self.fields.get(key)
        if f is None:
            f = flood(gmap, self.mask(gmap, fearless), fp, seeds)
            self.fields[key] = f
            self.floods += 1
        return f

    def prey_field(self, gmap, fp: int, prey,
                   fearless: bool = False) -> np.ndarray:
        """The prey field of footprint ``fp``: flooded from every living
        non-zombie in ``prey`` (their centres, in anchor space);
        ``fearless`` = over the mask without the fear of fire (the same as
        the plain field when ``fear_fire`` is off)."""
        s = self.settings
        fearless = bool(fearless and s is not None and s.fear_fire)
        key = ("prey", int(fp)) + (("fearless",) if fearless else ())
        return self._field(
            gmap, key, fp,
            [anchor_of(p.center_tile_y(), p.center_tile_x(), fp)
             for p in prey], fearless)

    def noise_field(self, gmap, fp: int, ty: int, tx: int) -> np.ndarray:
        """The field of a noise at tile ``(ty, tx)`` for footprint ``fp``
        (its anchor clamped into the map)."""
        h, w = gmap.material.shape
        sy, sx = anchor_of(ty, tx, fp)
        sy = min(max(sy, 0), h - 1)
        sx = min(max(sx, 0), w - 1)
        return self._field(gmap, ("noise", int(fp), sy, sx), fp, [(sy, sx)])


# ---------------------------------------------------------------------------
# The descent
# ---------------------------------------------------------------------------
def field_at(field: np.ndarray, y: int, x: int) -> int:
    """``field[y, x]``, :data:`UNREACHED` outside the map."""
    h, w = field.shape
    return int(field[y, x]) if (0 <= y < h and 0 <= x < w) else UNREACHED


def candidates(field: np.ndarray, y: int, x: int) -> list:
    """The cells a zombie at anchor ``(y, x)`` may step to on ``field``:
    the in-bounds 4-neighbours strictly below its own cell's value (an
    out-of-bounds or unreached own cell counts as :data:`UNREACHED`), as
    ``(value, order, ny, nx)`` lowest first, ties in the cell's
    :data:`DESCENT_ORDER`."""
    h, w = field.shape
    cur = field_at(field, y, x)
    out = []
    for k, (dy, dx) in enumerate(DESCENT_ORDER[(x + y) & 1]):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            v = int(field[ny, nx])
            if v < cur:
                out.append((v, k, ny, nx))
    out.sort()
    return out


def bash_tile(gmap, bashable, y: int, x: int, fp: int):
    """The door tile a zombie strikes to enter the ``fp`` block at anchor
    ``(y, x)``: ``None`` unless every tile of the block that blocks it now
    is a bashable closed-door tile; then the first such tile, row-major."""
    from simulation.materials import MAT_DOOR_CLOSED
    h, w = gmap.material.shape
    if y < 0 or x < 0 or y + fp > h or x + fp > w:
        return None
    mob = gmap.materials.mobility
    first = None
    for yy in range(y, y + fp):
        for xx in range(x, x + fp):
            m = gmap.material[yy, xx]
            if mob[m] > 0:
                continue
            if bashable[yy, xx] and int(m) == MAT_DOOR_CLOSED:
                if first is None:
                    first = (yy, xx)
            else:
                return None
    return first


def descend(gmap, field: np.ndarray, y: int, x: int, fp: int, state=None):
    """The move a zombie at anchor ``(y, x)`` makes on ``field``: the first
    of :func:`candidates` it can enter NOW (``is_passable_block`` on the
    live map -- the field may be a refresh old -- and, with ``fear_fire``,
    no feared tile in the block), as ``("step", ny, nx)``; with
    ``bash_doors``, the first one blocked only by a bashable closed door, as
    ``("bash", ny, nx, (ty, tx))``; ``None`` = it waits. ``state`` (the
    cache whose field this is) carries the settings and the bashable tiles;
    ``None`` is the stage-1 law."""
    s = getattr(state, "settings", None)
    fear_q = s.fear_temp_q16 if (s is not None and s.fear_fire) else None
    bashable = (state.bashable if (s is not None and s.bash_doors)
                else None)
    for _v, _k, ny, nx in candidates(field, y, x):
        if fear_q is not None and block_feared(gmap, ny, nx, fp, fear_q):
            continue
        if gmap.is_passable_block(ny, nx, fp):
            return ("step", ny, nx)
        if bashable is not None:
            t = bash_tile(gmap, bashable, ny, nx, fp)
            if t is not None:
                return ("bash", ny, nx, t)
    return None


# ---------------------------------------------------------------------------
# The wander (stage 2)
# ---------------------------------------------------------------------------
def wander_due(unit_id: int, tick: int, period: int) -> bool:
    """Whether a dormant zombie takes its wander step this tick: once every
    ``period`` ticks, staggered by its id."""
    return (int(tick) + int(unit_id) * WANDER_STAGGER) % int(period) == 0


def wander_direction(unit_id: int, tick: int) -> tuple:
    """The wander's direction ``(dy, dx)`` for ``(unit_id, tick)``: one of
    :data:`NEIGHBOURS`, from the Philox block as a seedless hash -- the same
    pair always gives the same direction, and nothing is drawn."""
    from simulation.philox32 import philox32_4x32_10, philox32_below
    w = philox32_4x32_10((int(tick) & _M32, 0, WANDER_SALT,
                          int(unit_id) & _M32), (0, 0))[0]
    return NEIGHBOURS[philox32_below(w, 4)]


def wander_step(gmap, y: int, x: int, ay: int, ax: int, leash_r2: int,
                unit_id: int, tick: int):
    """The cell a dormant zombie at anchor ``(y, x)``, anchored at ``(ay,
    ax)``, tries this tick: the hashed direction when that stays within the
    leash; when the zombie is already outside it (a push), the neighbour
    nearest the anchor (strictly nearer, ties in :data:`NEIGHBOURS` order);
    else ``None``."""
    def d2(py, px):
        return (py - ay) * (py - ay) + (px - ax) * (px - ax)
    cur = d2(y, x)
    if cur > leash_r2:
        best = None
        for dy, dx in NEIGHBOURS:
            nd = d2(y + dy, x + dx)
            if nd < cur and (best is None or nd < best[0]):
                best = (nd, y + dy, x + dx)
        return None if best is None else (best[1], best[2])
    dy, dx = wander_direction(unit_id, tick)
    ny, nx = y + dy, x + dx
    if d2(ny, nx) > leash_r2:
        return None
    return (ny, nx)


# ---------------------------------------------------------------------------
# Per-unit state (stage 2)
# ---------------------------------------------------------------------------
def clear(u) -> None:
    """A unit's horde state back to none (conversion, death, waking): no
    noise target, no wander anchor."""
    u.zombie_noise_y = -1
    u.zombie_noise_x = -1
    u.zombie_noise_ticks = 0
    u.zombie_anchor_y = -1
    u.zombie_anchor_x = -1


def settle(u, settings: HordeSettings) -> None:
    """A noise-chasing zombie gives up and goes dormant WHERE IT STANDS: its
    noise is dropped and, with the wander on, its anchor moves here."""
    u.zombie_noise_y = -1
    u.zombie_noise_x = -1
    u.zombie_noise_ticks = 0
    if settings.wander:
        u.zombie_anchor_y = int(u.tile_y)
        u.zombie_anchor_x = int(u.tile_x)


__all__ = ["UNREACHED", "NEIGHBOURS", "DESCENT_ORDER", "WANDER_SALT",
           "WANDER_STAGGER", "BEHAVIOUR_SWITCHES", "HordeSettings",
           "horde_settings", "block_passable", "flood_reference", "flood",
           "walkable_tiles", "feared_tiles", "block_feared", "is_hull_door",
           "bashable_door_tiles", "anchor_of", "HordeState", "field_at",
           "candidates", "bash_tile", "descend", "wander_due",
           "wander_direction", "wander_step", "clear", "settle"]
