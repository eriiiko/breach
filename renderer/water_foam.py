"""THE water foam (water look arc #97, patch W2: foam with a life;
docs/water_look_design_2026-11-19.md W2 + its critique §9: B2, M5, M9, M10,
M11): the law's numpy TWIN. Pure numpy, pyray-free. ``shaders/water_foam.fs``
(the step) and ``shaders/water.fs`` (the draw) mirror it and are GL-checked
against it (``tests/gl_water_foam_check.py``); the properties are asserted
here (``tests/test_water_foam.py``).

WHAT IT IS. A foam AMOUNT ``F`` in [0, 1] per CELL, ``m`` = ``[graphics.water]
foam_cells_per_tile`` cells per tile on each axis, over the whole map (cell
``(i, j)``'s centre at ``((j + .5) / m, (i + .5) / m)`` tiles). It is
PICTURE STATE: never synced, never read by the simulation, reset when the
sim tick goes back (a replay seek) and when a level loads. It is MADE by
sources, CARRIED by the simulated flow and LOST by decay:

    one foam step (``foam_step_s`` on the SIM clock), cut into n substeps
    so a substep moves at most ONE cell (critique M10), each substep:
      1. ADVECT  F <- MacCormack(F, v)   (Selle et al. 2008, + the clamp)
      2. SOURCE  F <- F + S dt
      3. DECAY   F <- F exp(-dt / foam_decay_s)
      4. CLAMP   F <- clamp(F, 0, 1); 0 on a BLOCKED cell (dry or solid)

  * THE INPUTS are only the two per-tile textures W1 uploads once per tick
    (``u_water`` = (ripple, ripple_v, depth, 0), ``u_flow`` = (vx, vy, solid,
    previous tick's depth), RGBA16F): read at a cell centre with the
    tile-centred bilinear the GPU would use (``heat_shimmer.bilinear_tiles``,
    CLAMP), except BLOCKED, read at the cell's own tile: its depth is 0 or
    it is solid. The shader does this bilinear itself with ``texelFetch``,
    so the GPU and the twin compute the same numbers. ``u_water``'s A is
    W2's: the relative agitation, packed per tick (:func:`relative_agitation`).
  * THE FLOW is W1's effective flow (``water_surface.edge_flow``: faded to 0
    over the first ``flow_edge_depth_m``), never a second flow.
  * THE SCHEME (M10) is MacCormack + clamp, as the fire's flame field
    (``fire_vfx_ref.advect_flame``): a forward semi-Lagrangian trace, a
    backward trace of its result, corrected by half the round-trip error,
    REVERTED to the forward value where it leaves the min / max of the four
    taps the forward trace landed between. A tap that falls on a BLOCKED
    cell (or off the map) reads the traced cell's OWN value -- the backtrace
    is clamped to wet, non-solid cells, so foam neither leaks into a wall nor
    is eaten by one. With no source the MAXIMUM never rises (M9): every
    value is a convex mix of cell values or reverted to one, decay only
    shrinks it, and the clamp and the kill only lower it.
  * THE SOURCES (M11), each a rate (amount per second) with its own gain:
      - WHITEWATER: the Froude number ``Fr = |v| / sqrt(g h)`` (h floored at
        ``flow_edge_depth_m``) past ``foam_froude``: ``gain x min(Fr - thr,
        1)`` -- fast, shallow water breaks;
      - STEEP SURFACE: W1's strength term (``water_surface.
        strength_ratio``: the flow over ``flow_speed_ref`` or the agitation
        over ``agitation_ref``) PAST full strength: ``gain x e^2``, ``e =
        clamp(ratio - 1)`` -- water rougher than the fine surface can show
        breaks (a blast's waves, a flood over twice ``flow_speed_ref``). The
        agitation here is the RELATIVE one (:func:`relative_agitation`, the
        tile's ``ripple_v`` against its wet neighbours', packed per tick in
        ``u_water``'s A): a surface heaving as one does not break -- and the
        sim's ripple field does heave whole rooms as one (after the water
        studio's pool blast every room's ``ripple_v`` jumps to ~0.8 m/s
        uniformly, then sits at a frozen ~0.04 m/s), which ``|ripple_v|``
        would foam over from wall to wall;
      - ADVANCING EDGE: depth appearing where there was little, from the
        previous tick's depth channel: ``gain x r^2``, ``r = clamp((h -
        h_prev) / h)`` the share of the depth that arrived this tick (1 where
        the water is new; squared, so a standing pool's sub-percent sloshing
        makes none);
      - WALLS: fast water along a wall, from the solid channel (its bilinear
        read is the wall's nearness, up to 3/8 within half a tile of one):
        ``gain x near x (|v| / flow_speed_ref)^2``.
    There is no pour source: no water source exists in a level today (M11).
  * Dry: a cell whose tile dries out is killed by the wet mask at once.

THE DRAW (``shaders/water.fs``; :func:`draw_cover`). The amount is a
THRESHOLD on a bubble raft (``noise_bake.bake_foam_rgba``, a uniform field)
read at TWO sizes (``foam_bubble_m``), both riding W1's one flow map
(``water_surface.layer_slides``: never a second advector): the mean of the
two reads is triangular on [0, 1], so :func:`lace_threshold` is the exact
threshold that covers the fraction ``f = clamp(F / foam_cover)`` -- a sheet
at the top, lace as it thins, nothing at 0 (Parberry: foam breaks into lace,
it does not fade). Each of the flow map's two layers is thresholded on its
own and the two covers cross-faded (a mix of two uniforms before the
threshold would thin the lace mid-cycle). Zoomed out, the lace fades to its
mean ``f`` with W1's zoom fade. A closed sheet hides the floor (it is drawn
OVER the water, opaque), is lit as the floor is, kills the glint, and
darkens the floor beneath it by ``foam_shadow`` x f.

Credits (repo rule -- cite what a file implements):
  - Andrew Selle, Ronald Fedkiw, ByungMoon Kim, Yingjie Liu, Jarek
    Rossignac, "An Unconditionally Stable MacCormack Method", J. Sci.
    Comput. 35 (2008) -- MacCormack advection with the min / max limiter.
    The fire's references: docs/papers/fire_vfx_references.md.
  - Mary Yingst, Jennifer R. Alford, Ian Parberry, "Very Fast Real-Time
    Ocean Wave Foam Rendering Using Halftoning", GAMEON-NA 2011, pp. 27-34
    -- the foam amount as a THRESHOLD on a fixed pattern (their dither
    array; our bubble raft), so foam breaks up rather than fading
    (archived: docs/papers/yingst2011_ocean_foam_halftoning.pdf).
  - Ojeda and Susin's real-time river foam (the amount made, carried by the
    flow and lost per frame), via docs/presentation_graphics_plan_2026-11-15.md
    section 4.
"""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from .heat_shimmer import bilinear_tiles
from .water_surface import (FlowFrame, WaterSettings, drift_m, edge_flow,
                            layer_slides, relative_agitation, smoothstep,
                            strength_ratio)

G_MPS2 = 9.81                 # gravity, for the Froude number
FOAM_SUBSTEP_CAP = 4          # substeps per foam step at most (M10)
CFL_MAX_CELLS = 1.0           # a substep moves at most this many cells
FOAM_STEP_CAP = 8             # foam steps per frame at most (SimClock cap)
FOAM_EDGE = 0.04              # the lace threshold's half-width (bubble units)
FOAM_ALBEDO = 0.85            # the foam's diffuse albedo (white froth)
# The larger bubble size reads the texture shifted, so the two sizes never
# line up where both coordinates happen to coincide.
FOAM_LARGE_OFFSET = (0.37, 0.61)

Box = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# The cell grid and its inputs
# ---------------------------------------------------------------------------

def cell_centres(h_tiles: int, w_tiles: int, m: int):
    """(x_tiles, y_tiles), each (h m, w m): every cell's centre in tiles."""
    jj, ii = np.meshgrid(np.arange(int(w_tiles) * int(m), dtype=np.float64),
                         np.arange(int(h_tiles) * int(m), dtype=np.float64))
    return (jj + 0.5) / m, (ii + 0.5) / m


class CellInputs(NamedTuple):
    """Per cell, from the two per-tile textures: constant within a tick."""
    v: np.ndarray          # (H, W, 2) the effective flow, m/s (y south)
    blocked: np.ndarray    # (H, W) bool: the cell's tile is dry or solid
    source: np.ndarray     # (H, W) the source rate, amount per second


def cell_inputs(water: np.ndarray, flow: np.ndarray, m: int,
                s: WaterSettings) -> CellInputs:
    """The cells' inputs from the PACKED textures (``water`` (h, w, 4) =
    (ripple, ripple_v, depth, 0), ``flow`` (h, w, 4) = (vx, vy, solid,
    previous depth); the float16 arrays the GPU receives)."""
    water = np.asarray(water, np.float64)
    flow = np.asarray(flow, np.float64)
    h, w = water.shape[:2]
    xt, yt = cell_centres(h, w, m)

    def bl(plane):
        return bilinear_tiles(plane, xt, yt)
    depth = bl(water[..., 2])
    v = edge_flow(np.stack([bl(flow[..., 0]), bl(flow[..., 1])], -1),
                  depth, s)
    ti = np.clip(np.floor(yt).astype(np.int64), 0, h - 1)
    tj = np.clip(np.floor(xt).astype(np.int64), 0, w - 1)
    blocked = (water[ti, tj, 2] <= 0.0) | (flow[ti, tj, 2] > 0.5)
    src = sources(v, depth, bl(flow[..., 3]), bl(flow[..., 2]),
                  bl(water[..., 3]), s)
    return CellInputs(v, blocked, src)


def sources(v: np.ndarray, depth, prev_depth, near_wall, agitation,
            s: WaterSettings, split: bool = False):
    """The source rate (amount per second) of the four sources (module
    docstring): ``v`` the effective flow (..., 2) m/s, ``depth`` /
    ``prev_depth`` m (this tick's and the previous tick's, bilinear),
    ``near_wall`` the bilinear solid channel, ``agitation`` the bilinear
    relative agitation (:func:`relative_agitation`).
    ``split`` returns the four rates (froude, steep, edge, wall)."""
    depth = np.asarray(depth, np.float64)
    speed = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    fr = speed / np.sqrt(G_MPS2 * np.maximum(depth, s.flow_edge_depth_m))
    s_fr = s.foam_gain_froude * np.clip(fr - s.foam_froude, 0.0, 1.0)
    steep = np.clip(strength_ratio(v, agitation, s) - 1.0, 0.0, 1.0)
    s_steep = s.foam_gain_steep * steep * steep
    rise = np.maximum(depth - np.asarray(prev_depth, np.float64), 0.0)
    frac = np.clip(rise / np.maximum(depth, 1e-6), 0.0, 1.0)
    s_edge = s.foam_gain_edge * frac * frac
    sv = np.clip(speed / s.flow_speed_ref, 0.0, 1.0)
    s_wall = s.foam_gain_wall * np.asarray(near_wall, np.float64) * sv * sv
    if split:
        return s_fr, s_steep, s_edge, s_wall
    return s_fr + s_steep + s_edge + s_wall


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def substeps(v_max_mps: float, dt_s: float, m: int, tile_m: float) -> int:
    """How many substeps one foam step of ``dt_s`` takes so a substep moves
    at most :data:`CFL_MAX_CELLS` at the fastest flow ``v_max_mps``,
    capped at :data:`FOAM_SUBSTEP_CAP` (past it the displacement is clamped
    to one cell: the foam lags a flood faster than that, never jumps)."""
    cells = float(v_max_mps) * float(dt_s) * int(m) / float(tile_m)
    n = int(math.ceil(cells / CFL_MAX_CELLS - 1e-9))
    return min(max(n, 1), FOAM_SUBSTEP_CAP)


def cell_step(v: np.ndarray, dt_s: float, m: int, tile_m: float
              ) -> np.ndarray:
    """A substep's displacement (..., 2) in CELLS: ``v dt m / tile_m``, its
    length clamped to :data:`CFL_MAX_CELLS`."""
    d = np.asarray(v, np.float64) * (float(dt_s) * int(m) / float(tile_m))
    ln = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    k = np.where(ln > CFL_MAX_CELLS, CFL_MAX_CELLS / np.maximum(ln, 1e-30),
                 1.0)
    return d * k[..., None]


def _traced(F: np.ndarray, blocked: np.ndarray, px: np.ndarray,
            py: np.ndarray, own: np.ndarray):
    """Bilinear read of ``F`` at cell coordinates (px, py) (cell centres at
    integers), a tap on a blocked cell or off the grid reading ``own``
    (the traced cell's own value); also the four taps' min and max."""
    h, w = F.shape
    x0 = np.floor(px)
    y0 = np.floor(py)
    fx, fy = px - x0, py - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    vals = []
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xc = np.clip(xi, 0, w - 1)
            yc = np.clip(yi, 0, h - 1)
            ok = inside & ~blocked[yc, xc]
            vals.append(np.where(ok, F[yc, xc], own))
    v00, v10, v01, v11 = vals
    val = ((v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy)
           + (v01 * (1.0 - fx) + v11 * fx) * fy)
    lo = np.minimum(np.minimum(v00, v10), np.minimum(v01, v11))
    hi = np.maximum(np.maximum(v00, v10), np.maximum(v01, v11))
    return val, lo, hi


def advect(F: np.ndarray, d_cells: np.ndarray, blocked: np.ndarray
           ) -> np.ndarray:
    """MacCormack + clamp (module docstring), ``d_cells`` (H, W, 2) each
    cell's displacement this substep. A blocked cell's own value is not
    computed here (the step kills it)."""
    F = np.asarray(F, np.float64)
    h, w = F.shape
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    fwd, lo, hi = _traced(F, blocked, jj - d_cells[..., 0],
                          ii - d_cells[..., 1], F)
    back, _, _ = _traced(fwd, blocked, jj + d_cells[..., 0],
                         ii + d_cells[..., 1], fwd)
    mc = fwd + 0.5 * (F - back)
    return np.where((mc < lo) | (mc > hi), fwd, mc)


def substep(F: np.ndarray, inp: CellInputs, dt_s: float, m: int,
            tile_m: float, s: WaterSettings) -> np.ndarray:
    """One substep: advect, add the sources, decay, clamp, kill."""
    F = advect(F, cell_step(inp.v, dt_s, m, tile_m), inp.blocked)
    F = F + inp.source * dt_s
    F = F * math.exp(-dt_s / s.foam_decay_s)
    F = np.clip(F, 0.0, 1.0)
    return np.where(inp.blocked, 0.0, F)


def step(F: np.ndarray, inp: CellInputs, v_max_mps: float, m: int,
         tile_m: float, s: WaterSettings) -> np.ndarray:
    """One foam step of ``foam_step_s``: :func:`substeps` equal substeps."""
    n = substeps(v_max_mps, s.foam_step_s, m, tile_m)
    dt = s.foam_step_s / n
    for _ in range(n):
        F = substep(F, inp, dt, m, tile_m, s)
    return F


def flow_max(flow: np.ndarray) -> float:
    """The fastest flow (m/s) in the packed flow texture: the bound every
    cell's bilinear, edge-faded flow lies under (per tick, not per frame)."""
    f = np.asarray(flow[..., :2], np.float32)
    return float(np.sqrt(float((f[..., 0] * f[..., 0]
                                + f[..., 1] * f[..., 1]).max())))


# ---------------------------------------------------------------------------
# The box the passes run over (cells), and what leaves it
# ---------------------------------------------------------------------------

def cell_box(tile_box: Optional[Box], m: int) -> Optional[Box]:
    """The wet tile box ``(x0, y0, x1, y1)`` (end-exclusive) in cells."""
    if tile_box is None:
        return None
    x0, y0, x1, y1 = tile_box
    return (x0 * m, y0 * m, x1 * m, y1 * m)


def leaving_rects(old: Optional[Box], new: Optional[Box]) -> List[Box]:
    """The part of ``old`` outside ``new`` as at most four rects: what a
    shrinking box must clear in BOTH buffers (the ping-pong's partial-pass
    trap). Everything outside the box is 0 in both, always."""
    if old is None:
        return []
    if new is None:
        return [old]
    ox0, oy0, ox1, oy1 = old
    nx0, ny0, nx1, ny1 = new
    ix0, iy0 = max(ox0, nx0), max(oy0, ny0)
    ix1, iy1 = min(ox1, nx1), min(oy1, ny1)
    if ix1 <= ix0 or iy1 <= iy0:
        return [old]
    out = []
    if oy0 < iy0:
        out.append((ox0, oy0, ox1, iy0))
    if iy1 < oy1:
        out.append((ox0, iy1, ox1, oy1))
    if ox0 < ix0:
        out.append((ox0, iy0, ix0, iy1))
    if ix1 < ox1:
        out.append((ix1, iy0, ox1, iy1))
    return out


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def cover_fraction(amount, s: WaterSettings) -> np.ndarray:
    """``f = clamp(F / foam_cover, 0, 1)``: the share of the surface the
    foam covers (and the zoomed-out cover, and the shadow's weight)."""
    return np.clip(np.asarray(amount, np.float64) / s.foam_cover, 0.0, 1.0)


def lace_threshold(f) -> np.ndarray:
    """The threshold ``t`` that the mean of two independent uniform bubble
    reads exceeds with probability ``f`` (its CDF is 2x^2 below 1/2 and
    1 - 2(1-x)^2 above), widened by :data:`FOAM_EDGE` on both ends so
    ``f = 1`` is a closed sheet and ``f = 0`` is nothing."""
    q = 1.0 - np.clip(np.asarray(f, np.float64), 0.0, 1.0)
    x = np.where(q < 0.5, np.sqrt(q * 0.5), 1.0 - np.sqrt((1.0 - q) * 0.5))
    return x * (1.0 + 2.0 * FOAM_EDGE) - FOAM_EDGE


def lace(b: np.ndarray, t) -> np.ndarray:
    """The lace's cover of bubble value ``b`` at threshold ``t``."""
    return smoothstep(0.0, 1.0, (np.asarray(b, np.float64) - (t - FOAM_EDGE))
                      / (2.0 * FOAM_EDGE))


def bubble_coords(x_m, y_m, d: np.ndarray, frame: FlowFrame,
                  s: WaterSettings):
    """One flow-map layer's bubble coordinates (texture units) at the two
    sizes, its slide ``d`` (..., 2) m from ``water_surface.layer_slides``."""
    from .noise_bake import FOAM_CELLS
    dx, dy = drift_m(frame, s)
    px = np.asarray(x_m, np.float64) - d[..., 0] - dx
    py = np.asarray(y_m, np.float64) - d[..., 1] - dy
    out = []
    for k, size in enumerate(s.foam_bubble_m):
        P = size * FOAM_CELLS
        ox, oy = FOAM_LARGE_OFFSET if k == 1 else (0.0, 0.0)
        out.append(np.stack([px / P + ox, py / P + oy], axis=-1))
    return out


def draw_cover(x_m, y_m, vel, depth, amount, frame: FlowFrame,
               tex: np.ndarray, s: WaterSettings) -> np.ndarray:
    """THE drawn foam cover in [0, 1] at world positions (metres): the
    module docstring's draw. ``vel`` (..., 2) the bilinear flow m/s,
    ``depth`` m, ``amount`` the bilinear foam amount, ``tex`` the bubble
    bake (read at level 0, as a magnified GPU read)."""
    from .water_surface import sample_bilinear_repeat
    d0, d1, w0, _v = layer_slides(x_m, y_m, vel, depth, frame, s)
    f = cover_fraction(amount, s)
    t = lace_threshold(f)
    covers = []
    for d in (d0, d1):
        qs, ql = bubble_coords(x_m, y_m, d, frame, s)
        b = 0.5 * (sample_bilinear_repeat(tex, qs)[..., 0]
                   + sample_bilinear_repeat(tex, ql)[..., 0])
        covers.append(lace(b, t))
    c = covers[1] * (1.0 - w0) + covers[0] * w0          # mix(c1, c0, w0)
    return f + (c - f) * frame.fade                      # mix(f, lace, fade)


def shadow_factor(amount, s: WaterSettings) -> np.ndarray:
    """What reaches the floor beneath the foam: ``1 - foam_shadow x f``."""
    return 1.0 - s.foam_shadow * cover_fraction(amount, s)


__all__ = [
    "G_MPS2", "FOAM_SUBSTEP_CAP", "CFL_MAX_CELLS", "FOAM_STEP_CAP",
    "FOAM_EDGE", "FOAM_ALBEDO", "FOAM_LARGE_OFFSET",
    "cell_centres", "CellInputs", "cell_inputs", "relative_agitation",
    "sources",
    "substeps", "cell_step", "advect", "substep", "step", "flow_max",
    "cell_box", "leaving_rects",
    "cover_fraction", "lace_threshold", "lace", "bubble_coords",
    "draw_cover", "shadow_factor",
]
