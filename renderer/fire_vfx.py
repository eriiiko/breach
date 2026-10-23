"""THE flame VFX pass, GPU side -- graphics arc G3 "Fire", patches F1
(substrate) and F2 (the loop).

Design: ``docs/g3_fire_design_2026-10-23.md`` (v3, APPROVED; §13 binding):
§3.2 fields, §3.4 loop + §13.1 screened solve + §13.3 clock, §5 + §13.10
active region, §9 + §13.11 ping-pong traps, §13.9 per-pass parity, §13.13 one
tamed wind, §13.14 per-tile sources.

WHAT IT IS. A render-only 2D fluid per tile column ("Layer B"), fed by the
game's ``fire``, the burning tile's temperature and THE tamed wind. Its numpy
SPEC TWIN is ``renderer/fire_vfx_ref.py``: the source law (``flame_sources``),
the cell rule (``rasterize_sources``), the puffing, the curl-noise modes and
every loop stage live THERE, and this module imports them -- it never
re-implements the source law. Each GL pass (``shaders/fire_vfx_<pass>.fs``) is
a NAMED COMPOSITION of twin stage functions (:data:`PASSES`), and the GL check
compares each pass with that composition one pass at a time (§13.9).

THE LOOP (F2), per fixed substep of ``dt_s`` on the active boxes:

    vel_advect  = advect_velocity -> follow_wind
    confine     = confine (vorticity confinement + curl noise x local I)
    divergence  = pressure_rhs  (div u* - q puff, into the rhs channel)
    jacobi x n  = jacobi_sweep  (screened, warm-started)
    project     = project       (subtract grad p, no-through)
    flame       = advect_flame (MacCormack + Selle + T limiter) -> inject
                  -> decay_cool

-- 5 + n_jacobi passes, exactly the twin's ``substep`` (the twin's dynamics
are untouched by F2; only pointwise neighbours were fused into one pass).

THE BUDGET (F2): every pass is DIRECT GL (``core.DirectPasses`` +
``PingPong.direct_pass``): ~4 us of CPU and ~3 us of fixed GPU cost per pass,
against ~40 / ~30 through raylib's batch (F1 measured; the batch's per-flush
VBO update serialised the GPU behind Python). That is what lets the loop keep
the twin's 1/120 s substep and 20 sweeps -- the look F0b tuned -- inside the
1 ms GPU / 1.5 ms CPU frame budget.

THE FIELDS: velocity (xy = u tiles/s, z = p, w = rhs) and flame (x = R, y =
R*T) -- two whole-map, world-anchored :class:`renderer.core.PingPong` fp32
fields on the SAME grid (``m`` cells per tile; ``m_u`` must equal ``m`` in
G3 -- ACCEPTED GAP: the laptop tier runs m = m_u = 4). Per-tile textures
(sources, flags + gas T, puffing phases, wind + local I) are uploaded once
per sim tick, for a WINDOW around the fire only (the cost scales with the
fire, not the map).

RENDER-ONLY. Reads read-only (``writeable=False``) dequantized copies of sim
fields, writes no synced state, never imports a sim writer, never runs
headless (the GL class imports pyray lazily; everything above it is pure
numpy and testable without a context). Float math is fine here (render layer,
determinism-exempt). Nothing is drawn into the picture yet (F3): only the
debug views (F11).

Credits: the techniques are the twin's (``renderer/fire_vfx_ref.py`` header;
archived in ``docs/papers/fire_vfx_references.md``) -- Stam 1999/2003, Fedkiw
et al. 2001, Nguyen et al. 2002, Harris 2004 (GPU Gems ch. 38, the
texture-grid ping-pong formulation used here), Selle et al. 2008, Bridson et
al. 2007, Cetegen & Ahmed 1993, Soerb's Ignitement breakdown (2026).
"""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import fire_vfx_ref as ref
from .fire_vfx_ref import FireVFXDials

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"
_COMMON_GLSL = "fire_vfx_common.glsl"
_INCLUDE_LINE = "// @include fire_vfx_common.glsl"

# Sampler units: above raylib's batch units (0..3), like GasReliefPass
# (renderer/gas_relief.py _FIRST_UNIT) -- raylib's batch binds at most four.
_FIRST_UNIT = 5
_UNIT = {"u_src_a": 5, "u_src_b": 6, "u_wind": 7, "u_puff": 8,
         "u_vel": 9, "u_flame": 10, "u_lut": 11}

LUT_SIZE = 1024                 # the blackbody GPU LUT (§3.5, §13.14)
MAX_BOXES = 4                   # active-region boxes (§5) == core.PASS_MAX_RECTS
SUBSTEP_CAP = 6                 # substeps per frame (§13.3: >= ceil(1/(tps dt)) + 1)
MAX_WALL_DT_S = 0.1             # an idle frame never counts more wall time than this
MAX_CURL_MODES = 32             # fire_vfx_confine.fs uniform arrays
PUFF_REBASE_S = 30.0            # re-anchor the puffing phases at least this often

# flags channel bits of the per-tile texture (fire_vfx_common.glsl FLAG_*)
FLAG_SOLID, FLAG_VACUUM = 1, 2

Box = Tuple[int, int, int, int]   # (x0, y0, x1, y1) tiles, end-exclusive


# ---------------------------------------------------------------------------
# Settings -- [render.fire_vfx] is THE tuning surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireVFXSettings:
    """Everything ``[render.fire_vfx]`` sets. ``dials`` are the loop's
    (``FireVFXDials``, the twin's schema); the rest are this pass's own."""
    enabled: bool = True
    m_u: int = 8                    # velocity/pressure cells per tile; == m in G3
    margin_tiles: int = 3           # active-region dilation (F0b: R spreads ~3 tiles)
    idle_when_paused: bool = True   # substeps keep running on frozen sources (§3.4)
    debug_view: int = 0             # 0 off, see DEBUG_VIEWS
    dials: FireVFXDials = field(default_factory=FireVFXDials)

    @property
    def m(self) -> int:
        return int(self.dials.m)


_OWN_KEYS = ("enabled", "m_u", "margin_tiles", "idle_when_paused", "debug_view")

# the F11 cycle (fire_vfx_debug.fs u_mode)
DEBUG_VIEWS = ("off", "sources", "flame R", "flame T_vfx", "velocity u",
               "pressure p")


def settings_from_config(cfg) -> FireVFXSettings:
    """``[render.fire_vfx]`` -> settings. SINGLE SOURCE: the config file is the
    live value of every dial; the dataclass defaults (``FireVFXDials``,
    ``FireVFXSettings``) are only the schema and the fallback for a missing
    key -- ``tests/test_fire_vfx.py`` asserts the shipped config names every
    one of them, so no fallback is silently live.

    ``m_u`` must equal ``m`` (the twin and the GL loop share one grid; a
    separate velocity grid is the laptop tier's ACCEPTED GAP for G3): a
    config that differs runs at ``m`` and says so once. ``curl_modes`` is
    capped at :data:`MAX_CURL_MODES` (the shader's uniform arrays)."""
    sec = getattr(getattr(cfg, "render", None), "fire_vfx", None)

    def get(name, default):
        if sec is None or not hasattr(sec, name):
            return default
        return type(default)(getattr(sec, name))

    dials = FireVFXDials(**{f.name: get(f.name, f.default)
                            for f in fields(FireVFXDials)})
    if dials.curl_modes > MAX_CURL_MODES:
        _warn_once("curl_modes", f"[render.fire_vfx] curl_modes "
                   f"{dials.curl_modes} > {MAX_CURL_MODES}: capped")
        dials = dials.with_(curl_modes=MAX_CURL_MODES)
    base = FireVFXSettings()
    own = {k: get(k, getattr(base, k)) for k in _OWN_KEYS}
    if own["m_u"] != dials.m:
        _warn_once("m_u", f"[render.fire_vfx] m_u {own['m_u']} != m "
                   f"{dials.m}: the loop runs both at m (G3 accepted gap)")
        own["m_u"] = dials.m
    return FireVFXSettings(dials=dials, **own)


_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[fire_vfx] WARN: {msg}")


def config_keys() -> Tuple[str, ...]:
    """Every key ``[render.fire_vfx]`` is expected to carry."""
    return _OWN_KEYS + tuple(f.name for f in fields(FireVFXDials))


# ---------------------------------------------------------------------------
# The sim clock (§13.3)
# ---------------------------------------------------------------------------

def clamp_alpha(alpha: Optional[float]) -> float:
    """The sub-tick fraction, clamped to [0, 1) (None = 0: a caller that
    passes no alpha steps on whole ticks)."""
    if alpha is None or not math.isfinite(float(alpha)):
        return 0.0
    return min(max(float(alpha), 0.0), math.nextafter(1.0, 0.0))


class SimClock:
    """The flame's clock: ``t_vfx = (tick + alpha) / tps``, never stepping
    back, cut into fixed substeps of ``dt_s``.

    :meth:`advance` returns how many substeps this frame runs: the substeps
    due on the sim clock since the last frame, plus (paused, idle on) those
    due on the wall-time idle counter ``t_idle`` -- which is never written
    into ``t_vfx`` -- capped at ``cap`` (a frame that owes more DROPS the
    rest: render-only, nothing to catch up). With one tick per frame (look
    shots, ``alpha = 0``) every frame runs exactly ``1 / (tps dt_s)``
    substeps, the same rate as the game at any frame rate (§13.3).

    A sim tick that goes BACKWARD (a new sim fed to the same renderer)
    re-anchors: ``t_vfx`` holds that frame and continues from there, so it
    stays monotone and never stalls waiting for the old tick to come back.
    """

    def __init__(self, dt_s: float, cap: int = SUBSTEP_CAP):
        self.dt_s = float(dt_s)
        self.cap = int(cap)
        self.t_vfx = 0.0
        self.t_idle = 0.0
        self._offset = 0.0
        self._raw_prev: Optional[float] = None
        self._k_sim: Optional[int] = None
        self._k_idle = 0
        self.dropped = 0
        self.last_n = 0

    def _k(self, t: float) -> int:
        return int(math.floor(t / self.dt_s + 1e-6))

    def set_dt(self, dt_s: float) -> None:
        if float(dt_s) != self.dt_s:
            self.dt_s = float(dt_s)
            self._k_sim = self._k(self.t_vfx)
            self._k_idle = self._k(self.t_idle)

    @property
    def t_noise(self) -> float:
        """The noise / puffing clock: ``t_vfx + t_idle`` (§13.3)."""
        return self.t_vfx + self.t_idle

    def advance(self, sim_tick: int, alpha: Optional[float], tps: float, *,
                paused: bool = False, idle_when_paused: bool = True,
                wall_dt: float = 0.0) -> int:
        raw = (float(sim_tick) + clamp_alpha(alpha)) / float(tps)
        if self._raw_prev is not None and raw < self._raw_prev:
            self._offset += self._raw_prev - raw
        self._raw_prev = raw
        self.t_vfx = max(self.t_vfx, raw + self._offset)
        k = self._k(self.t_vfx)
        if self._k_sim is None:
            self._k_sim = k                  # the first frame owes nothing
        n = k - self._k_sim
        self._k_sim = k
        if paused and idle_when_paused:
            self.t_idle += min(max(float(wall_dt), 0.0), MAX_WALL_DT_S)
            ki = self._k(self.t_idle)
            n += ki - self._k_idle
            self._k_idle = ki
        run = min(max(n, 0), self.cap)
        self.dropped += max(n, 0) - run
        self.last_n = run
        return run

    def substep_times(self, n: int) -> List[float]:
        """The noise clock at each of this frame's ``n`` substeps (the last
        one at :attr:`t_noise`)."""
        t = self.t_noise
        return [t - (n - 1 - j) * self.dt_s for j in range(n)]


# ---------------------------------------------------------------------------
# The active region (§5, §13.10)
# ---------------------------------------------------------------------------

def _clip_box(b: Box, w: int, h: int) -> Box:
    x0, y0, x1, y1 = b
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


def _area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _merge_overlapping(boxes: List[Box]) -> List[Box]:
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap(boxes[i], boxes[j]):
                    boxes[i] = _union(boxes[i], boxes[j])
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def cut_boxes(live: np.ndarray, margin_tiles: int,
              max_boxes: int = MAX_BOXES) -> List[Box]:
    """The active boxes: each 8-connected cluster of ``live`` (burning)
    tiles, its bounding box dilated by ``margin_tiles`` and clipped to the
    map; overlapping boxes merged; then, while there are more than
    ``max_boxes``, the pair whose union wastes the least area merged. The
    result is <= ``max_boxes`` DISJOINT boxes covering every live tile and
    its margin -- two distant fires never make one map-wide box (§5)."""
    live = np.asarray(live, dtype=bool)
    h, w = live.shape
    if not live.any():
        return []
    lab = ref._clusters(live)
    r = int(margin_tiles)
    boxes = []
    ly, lx = np.nonzero(live)
    keys = lab[ly, lx]
    for key in np.unique(keys):
        sel = keys == key
        ys, xs = ly[sel], lx[sel]
        boxes.append(_clip_box((int(xs.min()) - r, int(ys.min()) - r,
                                int(xs.max()) + 1 + r, int(ys.max()) + 1 + r),
                               w, h))
    boxes = _merge_overlapping(boxes)
    while len(boxes) > max_boxes:
        best = None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                u = _union(boxes[i], boxes[j])
                waste = _area(u) - _area(boxes[i]) - _area(boxes[j])
                if best is None or waste < best[0]:
                    best = (waste, i, j, u)
        _, i, j, u = best
        boxes = [b for k, b in enumerate(boxes) if k not in (i, j)] + [u]
        boxes = _merge_overlapping(boxes)
    return sorted(boxes)


def boxes_mask(boxes: Sequence[Box], shape: Tuple[int, int]) -> np.ndarray:
    """The tiles the boxes cover (the passes' DOMAIN this frame)."""
    m = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def mask_rects(mask: np.ndarray) -> List[Box]:
    """A mask as rectangles: horizontal runs per row, stacked down while the
    next row has the same run (few quads for the clear pass)."""
    mask = np.asarray(mask, dtype=bool)
    out: List[Box] = []
    if not mask.any():
        return out
    rows = np.nonzero(mask.any(axis=1))[0]
    open_runs = {}
    for y in range(int(rows[0]), int(rows[-1]) + 2):
        runs = set()
        if y < mask.shape[0] and mask[y].any():
            row = np.concatenate([[False], mask[y], [False]]).astype(np.int8)
            d = np.diff(row)
            runs = set(zip(np.nonzero(d == 1)[0].tolist(),
                           np.nonzero(d == -1)[0].tolist()))
        for run in list(open_runs):
            if run not in runs:
                y0 = open_runs.pop(run)
                out.append((run[0], y0, run[1], y))
        for run in runs:
            open_runs.setdefault(run, y)
    return out


def linger_s(dials: FireVFXDials) -> float:
    """How long a tile stays active after it stopped burning: the time the
    steady flame of a full-intensity source (R ~ s0 * lifetime * (1 +
    puff_amp)) takes to decay below ``r_cut`` -- so a fire that goes out
    FADES instead of being cut off by its region vanishing."""
    r_peak = dials.s0 * dials.lifetime_s * (1.0 + dials.puff_amp)
    return float(ref.extinction_time_s(dials, r_peak))


class ActiveRegion:
    """Per-tile bookkeeping of the region (pure numpy, no GL).

    ``update(burning, t)`` marks burning tiles live at time ``t`` (the noise
    clock), keeps them live for :func:`linger_s` after they stop, cuts the
    live tiles into boxes, and returns the tiles that LEFT the domain since
    the last update -- the ones to clear once (§5)."""

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(shape)
        self.last_burn = np.full(self.shape, -np.inf)
        self.boxes: List[Box] = []
        self.domain = np.zeros(self.shape, dtype=bool)

    def update(self, burning: np.ndarray, t: float, margin_tiles: int,
               linger: float) -> np.ndarray:
        burning = np.asarray(burning, dtype=bool)
        self.last_burn[burning] = t
        live = (t - self.last_burn) <= linger
        key = (int(margin_tiles), live.tobytes())
        if key == getattr(self, "_cut_key", None):
            return np.zeros(self.shape, dtype=bool)   # nothing moved
        self._cut_key = key
        self.boxes = cut_boxes(live, margin_tiles)
        new_domain = boxes_mask(self.boxes, self.shape)
        left = self.domain & ~new_domain
        self.domain = new_domain
        return left

    @property
    def active(self) -> bool:
        return bool(self.boxes)


# ---------------------------------------------------------------------------
# Per-tick tile inputs (the twin's source law, per tile -- §13.14)
# ---------------------------------------------------------------------------

def read_only(a) -> np.ndarray:
    """A ``writeable=False`` view: the VFX can read a sim array, never
    write it (design §8 'render-only, structurally')."""
    v = np.asarray(a).view()
    v.flags.writeable = False
    return v


def _bbox(mask: np.ndarray) -> Optional[Box]:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _dilate(b: Box, r: int, w: int, h: int) -> Box:
    return _clip_box((b[0] - r, b[1] - r, b[2] + r, b[3] + r), w, h)


def pack_window(candidates: np.ndarray, prior_boxes: Sequence[Box],
                margin_tiles: int) -> Optional[Box]:
    """The tile window a tick's pack covers (F2: the CPU cost scales with the
    fire, not the map). ``candidates`` = the tiles that may burn (``fire >
    i_min``, a superset of the source law's burning tiles); ``prior_boxes``
    = the active boxes before this frame's region update.

    The window = the candidates' bounding box dilated by ``margin + 1`` and
    every prior box dilated by 1, as one box, clipped to the map. It holds
    (a) every burning tile's whole cluster and its neighbours (the
    cropped-``flame_sources`` contract), (b) every box the region can cut
    this frame -- live tiles are burning now or were live before, and both
    dilate inside it -- and (c) one tile ring around them, which the
    passes' per-tile bilinear reads touch. ``None`` = nothing to pack."""
    h, w = candidates.shape
    parts = []
    bb = _bbox(candidates)
    if bb is not None:
        parts.append(_dilate(bb, int(margin_tiles) + 1, w, h))
    parts.extend(_dilate(b, 1, w, h) for b in prior_boxes)
    if not parts:
        return None
    out = parts[0]
    for b in parts[1:]:
        out = _union(out, b)
    return out


@dataclass
class TilePack:
    """One tick's per-tile textures for a WINDOW of the map (float32,
    (wh, ww, 4), numpy row 0 = the window's north row):

      ``src_a`` = (s 1/s, q 1/s, T_src K, cover code) -- ``flame_sources``;
      ``src_b`` = (puff wobble rates g1, g2 Hz, T_gas K, flags) -- flags =
                  solid 1 | vacuum 2;
      ``puff``  = (puff Hz, theta0, theta1, theta2 rad) -- the puffing's
                  phases at ``t_base`` (``fire_vfx_ref.puff_phases``);
      ``wind``  = (u_wind x, y tiles/s = tamed * tps, local I, 0).

    ``window`` = (x0, y0, x1, y1) tiles; ``burning`` is FULL-map (the
    region's input); ``hz`` / ``phase`` keep the per-tile puffing for a
    re-anchor (:meth:`rebase`)."""
    window: Box
    src_a: np.ndarray
    src_b: np.ndarray
    puff: np.ndarray
    wind: np.ndarray
    burning: np.ndarray
    hz: np.ndarray
    phase: np.ndarray
    t_base: float = 0.0
    sources: object = None          # the twin's FlameSources (window; tests, debug)

    def rebase(self, t_base: float) -> None:
        """Re-anchor the puffing phases at ``t_base`` (same function of t;
        keeps the GPU's float32 ``tau = t - t_base`` small)."""
        th0, th1, th2, _, _ = ref.puff_phases(self.hz, self.phase, t_base)
        self.puff[..., 1] = th0
        self.puff[..., 2] = th1
        self.puff[..., 3] = th2
        self.t_base = float(t_base)


def pack_tile_inputs(fire, t_game, solid, flammable, vacuum, wind_tamed, *,
                     tps: float, tile_size_m: float, scales,
                     dials: FireVFXDials, window: Optional[Box] = None,
                     t_base: float = 0.0) -> TilePack:
    """One tick's reads -> the per-tile textures of ``window`` (default: the
    whole map), through THE source law (``fire_vfx_ref.flame_sources``,
    cropped with map-invariant cluster keys) and THE canonical scales
    (game-T -> Kelvin, the ramp's glow floor). ``wind_tamed`` is the frame's
    one tamed wind (tiles per tick) or None when there is nothing burning.
    Every input is full-map and only read through the window's slices."""
    H, W = np.shape(fire)
    if window is None:
        window = (0, 0, W, H)
    x0, y0, x1, y1 = window
    sl = (slice(y0, y1), slice(x0, x1))
    tk = scales.to_kelvin(np.asarray(t_game)[sl])
    solid_w = np.asarray(solid, dtype=bool)[sl]
    vac_w = np.asarray(vacuum, dtype=bool)[sl]
    src = ref.flame_sources(np.asarray(fire)[sl], tk, solid_w,
                            np.asarray(flammable)[sl], vac_w,
                            t_amb_k=scales.t_amb_k,
                            glow_min_k=scales.glow_min_k,
                            tile_size_m=tile_size_m, dials=dials,
                            origin=(y0, x0), map_w=W)
    h, w = tk.shape
    th0, th1, th2, g1, g2 = ref.puff_phases(src.puff_hz, src.puff_phase, t_base)
    a = np.zeros((h, w, 4), dtype=np.float32)
    a[..., 0] = src.s
    a[..., 1] = src.q
    a[..., 2] = src.t_src_k
    a[..., 3] = src.cover
    b = np.zeros((h, w, 4), dtype=np.float32)
    b[..., 0] = g1
    b[..., 1] = g2
    b[..., 2] = tk
    b[..., 3] = solid_w * FLAG_SOLID + vac_w * FLAG_VACUUM
    pf = np.zeros((h, w, 4), dtype=np.float32)
    pf[..., 0] = src.puff_hz
    pf[..., 1] = th0
    pf[..., 2] = th1
    pf[..., 3] = th2
    wd = np.zeros((h, w, 4), dtype=np.float32)
    if wind_tamed is not None:
        wt = np.asarray(wind_tamed, dtype=np.float64)[sl] * float(tps)
        wd[..., 0] = wt[..., 0]
        wd[..., 1] = wt[..., 1]
    wd[..., 2] = ref.local_intensity(src.i_eff, dials.i_local_sigma_tiles)
    burning = np.zeros((H, W), dtype=bool)
    burning[sl] = src.burning
    return TilePack(window=tuple(window), src_a=a, src_b=b, puff=pf, wind=wd,
                    burning=burning, hz=src.puff_hz, phase=src.puff_phase,
                    t_base=float(t_base), sources=src)


# ---------------------------------------------------------------------------
# The blackbody GPU LUT (§3.5, §13.14)
# ---------------------------------------------------------------------------

def blackbody_lut_kelvins(ramp, n: int = LUT_SIZE) -> np.ndarray:
    """The LUT's sample points: ``n`` Kelvin evenly over the ramp's own
    domain ``[kelvin_floor, kelvin_ceil]`` (texel i at i / (n - 1))."""
    return np.linspace(ramp.kelvin_floor, ramp.kelvin_ceil, int(n))


def blackbody_lut_data(ramp, n: int = LUT_SIZE) -> np.ndarray:
    """``(1, n, 4)`` float32: rgb = the ramp's chroma (max channel 1), a = its
    HDR intensity, both ``BlackbodyRamp.emission_at_kelvin`` -- THE ramp,
    sampled finer; never a second T -> colour map. A shader reads texel
    ``(x (n - 1) + 0.5) / n`` with ``x = (K - floor) / (ceil - floor)``, so
    linear filtering interpolates between neighbouring samples (no banding)
    and the ends land on the ramp's own clamp."""
    chroma, inten = ramp.emission_at_kelvin(blackbody_lut_kelvins(ramp, n))
    out = np.zeros((1, int(n), 4), dtype=np.float32)
    out[0, :, :3] = chroma
    out[0, :, 3] = inten
    return out


# ---------------------------------------------------------------------------
# The passes (pure: what one substep and one frame issue)
# ---------------------------------------------------------------------------

# Each GL pass is a named composition of TWIN stage functions (in order) --
# the per-pass parity check (§13.9) compares the pass with exactly this.
PASSES: Dict[str, Tuple[str, ...]] = {
    "vel_advect": ("advect_velocity", "follow_wind"),
    "confine": ("confine",),
    "divergence": ("pressure_rhs",),
    "jacobi": ("jacobi_sweep",),
    "project": ("project",),
    "flame": ("advect_flame", "inject", "decay_cool"),
}
STAGES = tuple(PASSES)                         # one shader per pass
PASS_FIELD = {"vel_advect": "vel", "confine": "vel", "divergence": "vel",
              "jacobi": "vel", "project": "vel", "flame": "flame"}


def substep_passes(n_jacobi: int) -> List[str]:
    """One substep's passes, in the twin ``substep``'s order: 5 + n_jacobi."""
    return (["vel_advect", "confine", "divergence"]
            + ["jacobi"] * int(n_jacobi) + ["project", "flame"])


class PassItem:
    """One GL pass: ``kind`` 'clear' (write zeros into BOTH buffers of
    ``field`` on ``rects``) or 'stage' (one ping-pong pass of ``stage`` on
    the active ``rects`` at noise time ``t``). Rects in tiles."""
    __slots__ = ("kind", "field", "rects", "stage", "t")

    def __init__(self, kind: str, field: str, rects, stage: str = "",
                 t: float = 0.0):
        self.kind, self.field, self.rects = kind, field, rects
        self.stage, self.t = stage, t

    def __eq__(self, other):
        return (isinstance(other, PassItem)
                and (self.kind, self.field, self.rects, self.stage, self.t)
                == (other.kind, other.field, other.rects, other.stage, other.t))

    def __repr__(self):
        return f"PassItem({self.kind}, {self.field}, {self.stage}, t={self.t})"


def plan_frame(boxes: Sequence[Box], left_rects: Sequence[Box],
               substep_times: Sequence[float],
               n_jacobi: int = FireVFXDials.n_jacobi) -> List[PassItem]:
    """The passes of one frame: clear what left the region (once, both
    fields), then per substep the loop's passes on the active boxes -- each
    ONE pass whatever the box count (§13.10). No boxes and nothing left ->
    no passes at all."""
    plan: List[PassItem] = []
    if left_rects:
        for f in ("vel", "flame"):
            plan.append(PassItem("clear", f, list(left_rects)))
    if boxes:
        rects = list(boxes)
        seq = substep_passes(n_jacobi)
        for t in substep_times:
            for st in seq:
                plan.append(PassItem("stage", PASS_FIELD[st], rects,
                                     stage=st, t=float(t)))
    return plan


# ---------------------------------------------------------------------------
# The GL pass
# ---------------------------------------------------------------------------

def compose_shader_source(stage: str) -> str:
    """``shaders/fire_vfx_<stage>.fs`` with the common GLSL spliced in at its
    include line (also what tests/test_one_tonemap scans)."""
    src = (SHADERS_DIR / f"fire_vfx_{stage}.fs").read_text(encoding="utf-8")
    common = (SHADERS_DIR / _COMMON_GLSL).read_text(encoding="utf-8")
    if _INCLUDE_LINE not in src:
        raise ValueError(f"fire_vfx_{stage}.fs lacks '{_INCLUDE_LINE}'")
    return src.replace(_INCLUDE_LINE, common)


def _crc(a) -> int:
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8))


def boxes_uniform(boxes: Sequence[Box], m: int) -> List[float]:
    """The ``u_boxes[4]`` uniform: the boxes in CELLS (x0, y0, x1, y1),
    unused slots zero (an empty rect draws nothing and contains nothing)."""
    out = []
    for x0, y0, x1, y1 in list(boxes)[:MAX_BOXES]:
        out += [x0 * m, y0 * m, x1 * m, y1 * m]
    return out + [0.0] * (4 * MAX_BOXES - len(out))


class FireVFXPass:
    """Owns the flame VFX's GPU resources and runs its passes.

    Per frame, from ``GameRenderer.upload_state``: :meth:`update` (CPU --
    clock, per-tick source pack, active region, uploads, the frame plan).
    From ``compose_world`` before the scene RT is bound: :meth:`simulate`
    (GPU -- the plan's passes, one direct-GL block). In the display stage:
    :meth:`draw_debug`."""

    # uniforms each pass program may carry (absent ones are skipped)
    _UNIFORMS = ("u_grid", "u_m", "u_boxes", "u_target", "u_src_a", "u_src_b",
                 "u_wind", "u_puff", "u_vel", "u_flame", "u_k_follow", "u_dt",
                 "u_tau", "u_strip", "u_puff_amp", "u_puff_wobble",
                 "u_vort_eps", "u_curl_amp", "u_modes", "u_mode_phase",
                 "u_n_modes", "u_screen", "u_lifetime", "u_cool_rate",
                 "u_r_eps", "u_r_cut", "u_maccormack")
    _DEBUG_UNIFORMS = ("u_grid", "u_m", "u_mode", "u_src_a", "u_src_b",
                       "u_wind", "u_puff", "u_vel", "u_flame", "u_lut",
                       "u_lut_k", "u_lut_n", "u_strip", "u_s_vis", "u_r_vis",
                       "u_u_vis", "u_p_vis", "u_t_lo", "u_t_hi")

    def __init__(self, grid_h: int, grid_w: int, settings: FireVFXSettings,
                 ramp):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        self.h, self.w = int(grid_h), int(grid_w)
        self.settings = settings
        self.ramp = ramp
        self.debug_override: Optional[int] = None
        self.shaders = {}
        self._loc: Dict[str, Dict[str, int]] = {}
        for name in STAGES:
            sh = core.load_pass_shader(compose_shader_source(name))
            self.shaders[name] = sh
            self._loc[name] = {u: rl.get_shader_location(sh, u)
                               for u in self._UNIFORMS}
            for u, unit in _UNIT.items():           # samplers: fixed units
                loc = self._loc[name].get(u, -1)
                if loc >= 0:
                    rl.set_shader_value(sh, loc, rl.ffi.new("int[1]", [unit]),
                                        rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
        vs = str(SHADERS_DIR / "lighting.vs")
        self.debug_shader = core.load_shader_from_source(
            vs, compose_shader_source("debug"))
        self._dloc = {u: rl.get_shader_location(self.debug_shader, u)
                      for u in self._DEBUG_UNIFORMS}
        self._ucache: Dict[Tuple[str, str], object] = {}
        self._static_key: Dict[str, object] = {}
        # per-tile textures (texelFetch -> the filter never matters)
        self.src_a_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self.src_b_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self.puff_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self.wind_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self._pack: Optional[TilePack] = None
        self._uploaded_window: Optional[Box] = None
        # the blackbody GPU LUT
        self.lut_tex = core.create_dynamic_rgba32f_texture(LUT_SIZE, 1)
        rl.set_texture_filter(self.lut_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self._lut_key = None
        self.upload_lut(ramp)
        self.vel = self.flame = None
        self._alloc()
        self.clock = SimClock(settings.dials.dt_s)
        self._wall_prev: Optional[float] = None
        self._pack_key = None
        self._plan: List[PassItem] = []
        self._modes_key = None
        self._modes = None
        self.last_update_ms = 0.0
        self.last_simulate_ms = 0.0
        self.passes_last_frame = 0
        self.substeps_last_frame = 0

    # ---- resources -------------------------------------------------------

    def _alloc(self) -> None:
        core = self._core
        s = self.settings
        if self.vel is not None:
            self.vel.unload()
            self.flame.unload()
        self.vel = core.PingPong(self.w * s.m, self.h * s.m, fp32=True)
        self.flame = core.PingPong(self.w * s.m, self.h * s.m, fp32=True)
        self._grid_key = (s.m, s.m_u)
        self.region = ActiveRegion((self.h, self.w))
        self._uploaded_window = None

    @classmethod
    def from_config(cls, grid_h: int, grid_w: int, cfg, ramp) -> "FireVFXPass":
        return cls(grid_h, grid_w, settings_from_config(cfg), ramp)

    def upload_lut(self, ramp) -> None:
        """(Re)upload the LUT from the ramp (on a ramp change only)."""
        key = (ramp.kelvin_floor, ramp.kelvin_ceil, ramp.kelvin_glow_min,
               ramp.kelvin_ref, ramp.intensity_exponent, ramp.intensity_max)
        if key == self._lut_key:
            return
        self.ramp = ramp
        self._core.update_rgba32f_texture(self.lut_tex,
                                          blackbody_lut_data(ramp))
        self._lut_key = key

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    @property
    def debug_view(self) -> int:
        return int(self.debug_override if self.debug_override is not None
                   else self.settings.debug_view)

    def cycle_debug_view(self) -> int:
        self.debug_override = (self.debug_view + 1) % len(DEBUG_VIEWS)
        return self.debug_override

    @property
    def vram_bytes(self) -> int:
        tiles = 4 * self.w * self.h * 16 + LUT_SIZE * 16
        return self.vel.vram_bytes + self.flame.vram_bytes + tiles

    def refresh_settings(self, cfg) -> None:
        """Re-read ``[render.fire_vfx]`` (every frame: Ctrl+R is live).
        ``m`` reallocates the fields (zeroed); turning the pass off and on
        starts from empty fields. Every other dial is live next substep."""
        new = settings_from_config(cfg)
        was_on = self.settings.enabled
        self.settings = new
        if not new.enabled:
            self._plan = []
        if (new.m, new.m_u) != self._grid_key or (new.enabled and not was_on):
            self._alloc()
            self._pack_key = None
        self.clock.set_dt(new.dials.dt_s)

    def curl_modes(self):
        """The twin's curl-noise modes for the current dials (cached)."""
        d = self.settings.dials
        key = (d.seed, d.curl_modes, d.curl_wavelength_tiles, d.curl_rate_hz)
        if key != self._modes_key:
            self._modes = ref.curl_noise_modes(d)
            self._modes_key = key
        return self._modes

    # ---- per frame (CPU) -------------------------------------------------

    def update(self, *, fire, t_game, solid, flammable, vacuum,
               wind_fn: Callable[[], np.ndarray], sim_tick: int,
               alpha: Optional[float], paused: bool, tps: float,
               tile_size_m: float, scales) -> None:
        """Advance the clock, repack the per-tile inputs of the fire's
        window when the sim state changed (once per tick; CRC-keyed on the
        window so a paused edit shows too), update the active region and plan
        this frame's passes. Every array argument is read through a
        ``writeable=False`` view; ``wind_fn`` returns the frame's ONE tamed
        wind and is called only while something burns."""
        t0 = time.perf_counter()
        s = self.settings
        d = s.dials
        now = t0
        wall_dt = 0.0 if self._wall_prev is None else now - self._wall_prev
        self._wall_prev = now
        n = self.clock.advance(sim_tick, alpha, tps, paused=paused,
                               idle_when_paused=s.idle_when_paused,
                               wall_dt=wall_dt)
        fire = read_only(fire)
        cand = fire > d.i_min
        window = pack_window(cand, self.region.boxes, s.margin_tiles)
        repacked = False
        if window is None:
            self._pack = None
        else:
            x0, y0, x1, y1 = window
            sl = (slice(y0, y1), slice(x0, x1))
            key = (window, _crc(fire[sl]), _crc(np.asarray(t_game)[sl]),
                   _crc(np.asarray(solid)[sl]), _crc(np.asarray(vacuum)[sl]),
                   int(sim_tick), d, float(tps))
            if key != self._pack_key:
                self._pack_key = key
                any_fire = bool(cand[sl].any())
                self._pack = pack_tile_inputs(
                    fire, read_only(t_game), read_only(solid),
                    read_only(flammable), read_only(vacuum),
                    wind_fn() if any_fire else None, tps=tps,
                    tile_size_m=tile_size_m, scales=scales, dials=d,
                    window=window, t_base=self.clock.t_noise)
                repacked = True
            elif abs(self.clock.t_noise - self._pack.t_base) > PUFF_REBASE_S:
                self._pack.rebase(self.clock.t_noise)
                x0, y0, x1, y1 = self._pack.window
                self._upload_rect(self.puff_tex, self._pack.puff, x0, y0)
        burning = (self._pack.burning if self._pack is not None
                   else np.zeros((self.h, self.w), bool))
        left = self.region.update(burning, self.clock.t_noise,
                                  s.margin_tiles, linger_s(d))
        if repacked:
            self.upload_pack(self._pack)
        self._plan = plan_frame(self.region.boxes, mask_rects(left),
                                self.clock.substep_times(n)
                                if self.region.active else [], d.n_jacobi)
        self.substeps_last_frame = n if self.region.active else 0
        self.last_update_ms = (time.perf_counter() - t0) * 1000.0

    def _upload_rect(self, tex, arr: np.ndarray, x0: int, y0: int) -> None:
        rl = self._rl
        a = np.ascontiguousarray(arr, dtype=np.float32)
        h, w = a.shape[:2]
        rl.update_texture_rec(tex, rl.Rectangle(float(x0), float(y0),
                                                float(w), float(h)),
                              rl.ffi.cast("void *",
                                          rl.ffi.from_buffer("float[]", a)))

    def upload_pack(self, pack: TilePack) -> None:
        """Upload a pack's window into the per-tile textures. The rect sent
        is the union of this window and the last one, zero outside this
        window -- so no stale source survives where the fire was."""
        win = pack.window
        rect = win if self._uploaded_window is None else _union(
            win, self._uploaded_window)
        rx0, ry0, rx1, ry1 = rect
        x0, y0, x1, y1 = win
        for tex, arr in ((self.src_a_tex, pack.src_a),
                         (self.src_b_tex, pack.src_b),
                         (self.puff_tex, pack.puff),
                         (self.wind_tex, pack.wind)):
            if rect == win:
                self._upload_rect(tex, arr, x0, y0)
            else:
                full = np.zeros((ry1 - ry0, rx1 - rx0, 4), np.float32)
                full[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = arr
                self._upload_rect(tex, full, rx0, ry0)
        self._uploaded_window = win
        self._pack = pack

    # ---- per frame (GPU) -------------------------------------------------

    def _set(self, gp, sh: str, name: str, *v) -> None:
        """Set a uniform of the CURRENT program (``gp.use`` first) -- only
        when its value changed: a GL program keeps its uniforms."""
        loc = self._loc[sh].get(name, -1)
        if loc < 0:
            return
        ck = (sh, name)
        if self._ucache.get(ck) == v:
            return
        self._ucache[ck] = v
        gl = gp.gl
        if len(v) == 1:
            if isinstance(v[0], int) and not isinstance(v[0], bool):
                gl.Uniform1i(loc, int(v[0]))
            else:
                gl.Uniform1f(loc, float(v[0]))
        elif len(v) == 2:
            gl.Uniform2f(loc, float(v[0]), float(v[1]))
        else:
            raise ValueError(name)

    def _set_array(self, gp, sh: str, name: str, values, vec4: bool) -> None:
        loc = self._loc[sh].get(name, -1)
        if loc < 0:
            return
        ck = (sh, name)
        tv = tuple(float(x) for x in values)
        if self._ucache.get(ck) == tv:
            return
        self._ucache[ck] = tv
        arr = gp.gl.floats(tv)
        if vec4:
            gp.gl.Uniform4fv(loc, len(tv) // 4, arr)
        else:
            gp.gl.Uniform1fv(loc, len(tv), arr)

    def _program_uniforms(self, gp, st: str, t: float) -> None:
        """Every uniform of pass ``st`` at noise time ``t``. The STATIC ones
        (grid, boxes, dials) go to GL only when their key changed -- a GL
        program keeps its uniforms; the per-substep ones (the puffing's tau,
        the curl-noise phases) every substep."""
        d = self.settings.dials
        m = self.settings.m
        skey = (d, m, self._boxes_u, self.w, self.h, self._modes_key)
        if self._static_key.get(st) != skey:
            self._static_key[st] = skey
            self._set(gp, st, "u_grid", self.w, self.h)
            self._set(gp, st, "u_m", float(m))
            self._set(gp, st, "u_target", self.w * m, self.h * m)
            self._set_array(gp, st, "u_boxes", self._boxes_u, vec4=True)
            self._set(gp, st, "u_dt", d.dt_s)
            self._set(gp, st, "u_k_follow", d.k_follow)
            self._set(gp, st, "u_vort_eps", d.vort_eps)
            self._set(gp, st, "u_curl_amp", d.curl_amp)
            self._set(gp, st, "u_strip", d.strip_frac)
            self._set(gp, st, "u_puff_amp", d.puff_amp)
            self._set(gp, st, "u_puff_wobble", d.puff_wobble)
            self._set(gp, st, "u_screen", 1.0 / (m * m * d.screen_len_tiles ** 2))
            self._set(gp, st, "u_lifetime", d.lifetime_s)
            self._set(gp, st, "u_cool_rate", d.cool_rate)
            self._set(gp, st, "u_r_eps", d.r_eps)
            self._set(gp, st, "u_r_cut", d.r_cut)
            self._set(gp, st, "u_maccormack", int(bool(d.maccormack)))
            if st == "confine":
                kx, ky, _, _, amp = self.curl_modes()
                n = len(kx)
                self._set(gp, st, "u_n_modes", int(n))
                tab = np.zeros((MAX_CURL_MODES, 4))
                tab[:n, 0], tab[:n, 1], tab[:n, 2] = kx, ky, amp
                self._set_array(gp, st, "u_modes", tab.reshape(-1), vec4=True)
        if st in ("divergence", "flame"):
            # the puffing at tau = t - t_base: small however long the game runs
            self._set(gp, st, "u_tau",
                      t - (self._pack.t_base if self._pack is not None else 0.0))
        elif st == "confine":
            # float64 (omega t + phase) mod 2 pi: exact at any game age
            modes = self.curl_modes()
            gp.gl.Uniform1fv(self._loc[st]["u_mode_phase"], len(modes[0]),
                             gp.gl.floats(ref.curl_mode_phases(modes, t)))

    def _run_plan(self, plan: Sequence[PassItem]) -> None:
        """Issue a plan: the clears, then every stage pass in ONE direct-GL
        block. Per pass only what changed reaches GL: the program (when the
        stage changes), a stage's uniforms (once per stage and substep), the
        two field fronts (units 9 / 10) -- then one draw. This loop IS the
        CPU budget (F2: ~25 passes per substep)."""
        clears = [it for it in plan if it.kind == "clear"]
        m = self.settings.m
        for it in clears:
            pp = self.vel if it.field == "vel" else self.flame
            pp.clear_rects([(x0 * m, y0 * m, x1 * m, y1 * m)
                            for (x0, y0, x1, y1) in it.rects])
        stages = [it for it in plan if it.kind == "stage"]
        if not stages:
            return
        vel, flame = self.vel, self.flame
        n_rects = min(len(self.region.boxes), MAX_BOXES)
        self._boxes_u = tuple(float(v) for v in
                              boxes_uniform(self.region.boxes, m))
        self.curl_modes()                    # refresh the modes' cache key
        unit_v, unit_f = _UNIT["u_vel"], _UNIT["u_flame"]
        with self._core.DirectPasses() as gp:
            for name, tex in (("u_src_a", self.src_a_tex),
                              ("u_src_b", self.src_b_tex),
                              ("u_wind", self.wind_tex),
                              ("u_puff", self.puff_tex)):
                gp.bind(_UNIT[name], tex.id)
            bound_v = bound_f = None
            set_for: Dict[str, float] = {}
            for it in stages:
                st = it.stage
                gp.use(self.shaders[st])
                if set_for.get(st) != it.t:
                    self._program_uniforms(gp, st, it.t)
                    set_for[st] = it.t
                vid, fid = vel.front_id, flame.front_id
                if vid != bound_v:
                    gp.bind(unit_v, vid)
                    bound_v = vid
                if fid != bound_f:
                    gp.bind(unit_f, fid)
                    bound_f = fid
                (vel if PASS_FIELD[st] == "vel" else flame).direct_pass(
                    gp, n_rects)

    def run_pass(self, stage: str, t: float = 0.0) -> None:
        """Issue ONE pass of ``stage`` on the current region boxes at noise
        time ``t`` (public for the GL checks, §13.9)."""
        self._run_plan([PassItem("stage", PASS_FIELD[stage],
                                 list(self.region.boxes), stage=stage,
                                 t=float(t))])

    def simulate(self) -> int:
        """Run this frame's planned passes. Must run OUTSIDE any texture
        mode (each pass binds its own target): the renderer calls it before
        the scene RT is begun. Returns the pass count."""
        t0 = time.perf_counter()
        plan, self._plan = self._plan, []
        if plan:
            self._run_plan(plan)
        self.passes_last_frame = len(plan)
        self.last_simulate_ms = (time.perf_counter() - t0) * 1000.0
        return len(plan)

    # ---- debug (display stage) -------------------------------------------

    def _dset(self, name: str, *v) -> None:
        rl = self._rl
        loc = self._dloc.get(name, -1)
        if loc < 0:
            return
        if len(v) == 1 and isinstance(v[0], int):
            rl.set_shader_value(self.debug_shader, loc,
                                rl.ffi.new("int[1]", [int(v[0])]),
                                rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
            return
        kind = {1: rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
                2: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2}[len(v)]
        rl.set_shader_value(self.debug_shader, loc,
                            rl.ffi.new(f"float[{len(v)}]",
                                       [float(x) for x in v]), kind)

    def draw_debug(self, world_px_w: int, world_px_h: int) -> None:
        """The debug view over the tone-mapped picture (display space; dev
        only, F11 cycles, ``debug_view`` seeds): 1 the sources, 2 the flame
        density R, 3 its temperature T_vfx (blackbody LUT), 4 the velocity u
        (hue = direction, value = speed), 5 the pressure p (red +, blue -).
        Every view outlines the active boxes."""
        mode = self.debug_view
        if mode == 0:
            return
        rl = self._rl
        d = self.settings.dials
        if self.region.active and self._pack is not None:
            self._dset("u_grid", self.w, self.h)
            self._dset("u_m", float(self.settings.m))
            self._dset("u_mode", int(mode))
            self._dset("u_lut_k", self.ramp.kelvin_floor, self.ramp.kelvin_ceil)
            self._dset("u_lut_n", float(LUT_SIZE))
            self._dset("u_strip", d.strip_frac)
            self._dset("u_s_vis", d.s0)
            self._dset("u_r_vis", d.s0 * d.lifetime_s)
            self._dset("u_u_vis", 4.0)
            self._dset("u_p_vis", 0.05)
            self._dset("u_t_lo", float(self.ramp.kelvin_glow_min) - 200.0)
            self._dset("u_t_hi", float(self.ramp.kelvin_glow_min) + 600.0)
            rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA)
            rl.begin_shader_mode(self.debug_shader)
            samplers = (("u_src_a", self.src_a_tex.id),
                        ("u_src_b", self.src_b_tex.id),
                        ("u_wind", self.wind_tex.id),
                        ("u_puff", self.puff_tex.id),
                        ("u_vel", self.vel.front.id),
                        ("u_flame", self.flame.front.id),
                        ("u_lut", self.lut_tex.id))
            for name, tex_id in samplers:
                rl.rl_active_texture_slot(_UNIT[name])
                rl.rl_enable_texture(tex_id)
                self._dset(name, _UNIT[name])
            rl.rl_active_texture_slot(0)
            rl.draw_texture_pro(
                self.src_a_tex, rl.Rectangle(0, 0, float(self.w), float(self.h)),
                rl.Rectangle(0, 0, float(world_px_w), float(world_px_h)),
                rl.Vector2(0, 0), 0.0, rl.WHITE)
            rl.end_shader_mode()
            for name, _ in samplers:
                rl.rl_active_texture_slot(_UNIT[name])
                rl.rl_disable_texture()
            rl.rl_active_texture_slot(0)
            rl.end_blend_mode()
        ppx = float(world_px_w) / self.w
        ppy = float(world_px_h) / self.h
        for x0, y0, x1, y1 in self.region.boxes:
            rl.draw_rectangle_lines_ex(
                rl.Rectangle(x0 * ppx, y0 * ppy, (x1 - x0) * ppx,
                             (y1 - y0) * ppy),
                max(2.0, ppx / 16.0), rl.Color(80, 220, 255, 220))

    def unload(self) -> None:
        rl = self._rl
        for sh in self.shaders.values():
            rl.unload_shader(sh)
        rl.unload_shader(self.debug_shader)
        for t in (self.src_a_tex, self.src_b_tex, self.puff_tex,
                  self.wind_tex, self.lut_tex):
            rl.unload_texture(t)
        self.vel.unload()
        self.flame.unload()


__all__ = [
    "FireVFXSettings", "settings_from_config", "config_keys", "clamp_alpha",
    "SimClock", "cut_boxes", "boxes_mask", "mask_rects", "linger_s",
    "ActiveRegion", "read_only", "pack_window", "TilePack", "pack_tile_inputs",
    "blackbody_lut_kelvins", "blackbody_lut_data", "PASSES", "STAGES",
    "PASS_FIELD", "substep_passes", "PassItem", "plan_frame",
    "boxes_uniform", "compose_shader_source", "FireVFXPass", "DEBUG_VIEWS",
    "LUT_SIZE", "MAX_BOXES", "SUBSTEP_CAP", "MAX_CURL_MODES",
    "FLAG_SOLID", "FLAG_VACUUM",
]
