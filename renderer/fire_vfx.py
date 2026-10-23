"""THE flame VFX pass, GPU side -- graphics arc G3 "Fire", patch F1 (substrate).

Design: ``docs/g3_fire_design_2026-10-23.md`` (v3, APPROVED; §13 binding):
§3.2 fields, §3.4 loop + §13.3 clock, §5 + §13.10 active region, §9 + §13.11
ping-pong traps, §13.13 one tamed wind, §13.14 per-tile sources.

WHAT IT IS. A render-only 2D fluid per tile column ("Layer B"), fed by the
game's ``fire``, the burning tile's temperature and THE tamed wind. Its numpy
SPEC TWIN is ``renderer/fire_vfx_ref.py``: the source law (``flame_sources``),
the cell rule (``rasterize_sources``), the puffing (``puff_factor``) and every
loop stage live THERE, and this module imports them -- it never re-implements
the source law. Each GL stage shader (``shaders/fire_vfx_<stage>.fs``) is the
twin's function of the same name, one pass each, so F2 checks them one stage
at a time (§13.9).

F1 = THE SUBSTRATE, nothing drawn in the picture yet:
  * the field RTs: velocity + pressure at ``m_u`` cells per tile, flame density
    R and premultiplied temperature R*T at ``m`` -- whole-map, world-anchored,
    each a :class:`renderer.core.PingPong` (fp32 when rlgl's FBO takes it);
  * the per-tile source / flags / wind textures, uploaded once per sim tick
    from the twin's source law (nearest per tile, no CPU upsampling, §13.14);
  * the blackbody GPU LUT (1024 entries, linear filtering) from
    ``BlackbodyRamp.emission_at_kelvin`` -- uploaded HERE so ``blackbody.py``
    stays pyray-free (M5);
  * the active region (§5, §13.10): burning tiles dilated by ``margin_tiles``,
    cut into <= 4 boxes; every stage is ONE pass with up to 4 quads; tiles
    leaving the region are cleared once in both buffers; no fire -> no passes;
  * the sim clock (§13.3): ``t_vfx = (tick + alpha) / tps``, fixed substeps of
    ``dt_s``, capped at 6 per frame (a shot frame = one tick = exactly 5);
    ``idle_when_paused`` runs on its own wall counter ``t_idle``;
  * a pass chain that proves the plumbing with the twin's three POINTWISE
    stages: ``follow_wind`` (velocity), ``inject`` and ``decay_cool`` (flame).
    The real loop (advect / confine / screened solve / project) is F2;
  * a debug view (sources, or the R the passes made, + the active boxes),
    display-space like the other debug colormaps.

RENDER-ONLY. Reads read-only (``writeable=False``) dequantized copies of sim
fields, writes no synced state, never imports a sim writer, never runs
headless (the GL class imports pyray lazily; everything above it is pure
numpy and testable without a context). Float math is fine here (render layer,
determinism-exempt).

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
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import fire_vfx_ref as ref
from .fire_vfx_ref import FireVFXDials

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"
_COMMON_GLSL = "fire_vfx_common.glsl"
_INCLUDE_LINE = "// @include fire_vfx_common.glsl"

# Sampler units: above raylib's batch units (0..4), like GasReliefPass
# (renderer/gas_relief.py _FIRST_UNIT) -- raylib's batch binds at most four.
_FIRST_UNIT = 5

LUT_SIZE = 1024                 # the blackbody GPU LUT (§3.5, §13.14)
MAX_BOXES = 4                   # active-region boxes (§5)
SUBSTEP_CAP = 6                 # substeps per frame (§13.3: >= ceil(1/(tps dt)) + 1)
MAX_WALL_DT_S = 0.1             # an idle frame never counts more wall time than this

# flags channel bits of the per-tile texture (fire_vfx_common.glsl FLAG_*)
FLAG_SOLID, FLAG_VACUUM, FLAG_DOMAIN = 1, 2, 4

Box = Tuple[int, int, int, int]   # (x0, y0, x1, y1) tiles, end-exclusive


# ---------------------------------------------------------------------------
# Settings -- [render.fire_vfx] is THE tuning surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireVFXSettings:
    """Everything ``[render.fire_vfx]`` sets. ``dials`` are the loop's
    (``FireVFXDials``, the twin's schema); the rest are this pass's own."""
    enabled: bool = True
    m_u: int = 8                    # velocity/pressure cells per tile (laptop 4)
    margin_tiles: int = 3           # active-region dilation (F0b: R spreads ~3 tiles)
    idle_when_paused: bool = True   # substeps keep running on frozen sources (§3.4)
    debug_view: int = 0             # 0 off, 1 sources, 2 flame R (+ boxes)
    dials: FireVFXDials = field(default_factory=FireVFXDials)

    @property
    def m(self) -> int:
        return int(self.dials.m)


_OWN_KEYS = ("enabled", "m_u", "margin_tiles", "idle_when_paused", "debug_view")


def settings_from_config(cfg) -> FireVFXSettings:
    """``[render.fire_vfx]`` -> settings. SINGLE SOURCE: the config file is the
    live value of every dial; the dataclass defaults (``FireVFXDials``,
    ``FireVFXSettings``) are only the schema and the fallback for a missing
    key -- ``tests/test_fire_vfx.py`` asserts the shipped config names every
    one of them, so no fallback is silently live."""
    sec = getattr(getattr(cfg, "render", None), "fire_vfx", None)

    def get(name, default):
        if sec is None or not hasattr(sec, name):
            return default
        return type(default)(getattr(sec, name))

    dials = FireVFXDials(**{f.name: get(f.name, f.default)
                            for f in fields(FireVFXDials)})
    base = FireVFXSettings()
    own = {k: get(k, getattr(base, k)) for k in _OWN_KEYS}
    return FireVFXSettings(dials=dials, **own)


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
    for key in np.unique(lab[lab >= 0]):
        ys, xs = np.nonzero(lab == key)
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
    open_runs = {}
    for y in range(mask.shape[0] + 1):
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


@dataclass
class TilePack:
    """One tick's per-tile textures (float32, (h, w, 4), numpy row 0 =
    north = GL row 0):

      ``src_a`` = (s 1/s, q 1/s, T_src K, cover code) -- ``flame_sources``;
      ``src_b`` = (puff Hz, puff phase, T_gas K, flags) -- flags = solid 1 |
                  vacuum 2 | domain 4 (the domain bit is set per frame);
      ``wind``  = (u_wind x, y tiles/s = tamed * tps, local I, 0).
    """
    src_a: np.ndarray
    src_b: np.ndarray
    wind: np.ndarray
    burning: np.ndarray
    sources: object = None          # the twin's FlameSources (tests, debug)


def _local_intensity_cropped(i_eff: np.ndarray, sigma: float) -> np.ndarray:
    """``ref.local_intensity`` evaluated on the burning tiles' bounding box
    plus its reach only (exact: every tile outside is beyond the reach of
    every burning tile, so it is 0 either way)."""
    out = np.zeros(i_eff.shape)
    ys, xs = np.nonzero(i_eff > 0)
    if ys.size == 0:
        return out
    r = int(np.ceil(2.0 * sigma))
    h, w = i_eff.shape
    y0, y1 = max(0, ys.min() - 2 * r), min(h, ys.max() + 1 + 2 * r)
    x0, x1 = max(0, xs.min() - 2 * r), min(w, xs.max() + 1 + 2 * r)
    out[y0:y1, x0:x1] = ref.local_intensity(i_eff[y0:y1, x0:x1], sigma)
    return out


def pack_tile_inputs(fire, t_game, solid, flammable, vacuum, wind_tamed, *,
                     tps: float, tile_size_m: float, scales,
                     dials: FireVFXDials) -> TilePack:
    """One tick's reads -> the per-tile textures, through THE source law
    (``fire_vfx_ref.flame_sources``) and THE canonical scales (game-T ->
    Kelvin, the ramp's glow floor). ``wind_tamed`` is the frame's one tamed
    wind (tiles per tick) or None when there is nothing burning."""
    tk = scales.to_kelvin(t_game)
    solid = np.asarray(solid, dtype=bool)
    vacuum = np.asarray(vacuum, dtype=bool)
    src = ref.flame_sources(fire, tk, solid, flammable, vacuum,
                            t_amb_k=scales.t_amb_k,
                            glow_min_k=scales.glow_min_k,
                            tile_size_m=tile_size_m, dials=dials)
    h, w = tk.shape
    a = np.zeros((h, w, 4), dtype=np.float32)
    a[..., 0] = src.s
    a[..., 1] = src.q
    a[..., 2] = src.t_src_k
    a[..., 3] = src.cover
    b = np.zeros((h, w, 4), dtype=np.float32)
    b[..., 0] = src.puff_hz
    b[..., 1] = src.puff_phase
    b[..., 2] = tk
    b[..., 3] = solid * FLAG_SOLID + vacuum * FLAG_VACUUM
    wd = np.zeros((h, w, 4), dtype=np.float32)
    if wind_tamed is not None:
        wt = np.asarray(wind_tamed, dtype=np.float64) * float(tps)
        wd[..., 0] = wt[..., 0]
        wd[..., 1] = wt[..., 1]
    wd[..., 2] = _local_intensity_cropped(src.i_eff, dials.i_local_sigma_tiles)
    return TilePack(src_a=a, src_b=b, wind=wd, burning=src.burning,
                    sources=src)


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
# The frame plan (pure: which passes this frame issues)
# ---------------------------------------------------------------------------

STAGES = ("follow_wind", "inject", "decay_cool")   # F1's pointwise chain
STAGE_FIELD = {"follow_wind": "vel", "inject": "flame", "decay_cool": "flame"}


@dataclass
class PassItem:
    """One GL pass: ``kind`` 'clear' (write zeros into BOTH buffers of
    ``field`` on ``rects``) or 'stage' (one ping-pong pass of ``stage`` on
    the active ``rects`` at noise time ``t``). Rects in tiles."""
    kind: str
    field: str
    rects: List[Box]
    stage: str = ""
    t: float = 0.0


def plan_frame(boxes: Sequence[Box], left_rects: Sequence[Box],
               substep_times: Sequence[float]) -> List[PassItem]:
    """The passes of one frame: clear what left the region (once, both
    fields), then per substep the stage chain on the active boxes -- each
    stage ONE pass whatever the box count (§13.10). No boxes and nothing
    left -> no passes at all."""
    plan: List[PassItem] = []
    if left_rects:
        for f in ("vel", "flame"):
            plan.append(PassItem("clear", f, list(left_rects)))
    if boxes:
        for t in substep_times:
            for st in STAGES:
                plan.append(PassItem("stage", STAGE_FIELD[st], list(boxes),
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


class FireVFXPass:
    """Owns the flame VFX's GPU resources and runs its passes.

    Per frame, from ``GameRenderer.upload_state``: :meth:`update` (CPU --
    clock, per-tick source pack, active region, uploads, the frame plan).
    From ``compose_world`` before the scene RT is bound: :meth:`simulate`
    (GPU -- the plan's passes). In the display stage: :meth:`draw_debug`."""

    _UNIFORMS = ("u_grid", "u_m", "u_src_a", "u_src_b", "u_wind", "u_self",
                 "u_k_follow", "u_dt", "u_t", "u_strip", "u_puff_amp",
                 "u_puff_wobble", "u_lifetime", "u_cool_rate", "u_r_eps",
                 "u_r_cut", "u_mode", "u_flame", "u_lut", "u_lut_k",
                 "u_lut_n", "u_s_vis", "u_r_vis")

    def __init__(self, grid_h: int, grid_w: int, settings: FireVFXSettings,
                 ramp):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self.h, self.w = int(grid_h), int(grid_w)
        self.settings = settings
        self.ramp = ramp
        self.debug_override: Optional[int] = None
        vs = str(SHADERS_DIR / "lighting.vs")
        self.shaders = {}
        self._loc = {}
        for name in STAGES + ("debug",):
            sh = core.load_shader_from_source(vs, compose_shader_source(name))
            self.shaders[name] = sh
            self._loc[name] = {u: rl.get_shader_location(sh, u)
                               for u in self._UNIFORMS}
        # per-tile textures (texelFetch -> the filter never matters)
        self.src_a_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self.src_b_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self.wind_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        self._pack: Optional[TilePack] = None
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
        self.vel = core.PingPong(self.w * s.m_u, self.h * s.m_u, fp32=True)
        self.flame = core.PingPong(self.w * s.m, self.h * s.m, fp32=True)
        self._grid_key = (s.m, s.m_u)
        self.region = ActiveRegion((self.h, self.w))

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
        self.debug_override = (self.debug_view + 1) % 3
        return self.debug_override

    @property
    def vram_bytes(self) -> int:
        tiles = 3 * self.w * self.h * 16 + LUT_SIZE * 16
        return self.vel.vram_bytes + self.flame.vram_bytes + tiles

    def refresh_settings(self, cfg) -> None:
        """Re-read ``[render.fire_vfx]`` (every frame: Ctrl+R is live).
        ``m`` / ``m_u`` reallocate the fields (zeroed); turning the pass off
        and on starts from empty fields."""
        new = settings_from_config(cfg)
        was_on = self.settings.enabled
        self.settings = new
        if not new.enabled:
            self._plan = []
        if (new.m, new.m_u) != self._grid_key or (new.enabled and not was_on):
            self._alloc()
            self._pack_key = None
        self.clock.set_dt(new.dials.dt_s)

    # ---- per frame (CPU) -------------------------------------------------

    def update(self, *, fire, t_game, solid, flammable, vacuum,
               wind_fn: Callable[[], np.ndarray], sim_tick: int,
               alpha: Optional[float], paused: bool, tps: float,
               tile_size_m: float, scales) -> None:
        """Advance the clock, repack the per-tile inputs when the sim state
        changed (once per tick; CRC-keyed so a paused edit shows too),
        update the active region and plan this frame's passes. Every array
        argument is read through a ``writeable=False`` view; ``wind_fn``
        returns the frame's ONE tamed wind and is called only while
        something burns."""
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
        key = (_crc(fire), _crc(t_game), _crc(solid), _crc(vacuum),
               int(sim_tick), d, float(tps))
        repacked = False
        if key != self._pack_key:
            self._pack_key = key
            any_fire = bool((fire > d.i_min).any())
            # no fire and no region: nothing to pack (the zero-cost path)
            if any_fire or self.region.active:
                self._pack = pack_tile_inputs(
                    fire, read_only(t_game), read_only(solid),
                    read_only(flammable), read_only(vacuum),
                    wind_fn() if any_fire else None, tps=tps,
                    tile_size_m=tile_size_m, scales=scales, dials=d)
                repacked = True
        burning = (self._pack.burning if self._pack is not None
                   else np.zeros((self.h, self.w), bool))
        boxes_before = self.region.boxes
        left = self.region.update(burning, self.clock.t_noise,
                                  s.margin_tiles, linger_s(d))
        if self.region.active and (repacked
                                   or self.region.boxes != boxes_before):
            dom = self.region.domain
            fl = self._pack.src_b[..., 3].astype(np.int32) & ~FLAG_DOMAIN
            self._pack.src_b[..., 3] = fl | (dom * FLAG_DOMAIN)
            core = self._core
            core.update_rgba32f_texture(self.src_a_tex, self._pack.src_a)
            core.update_rgba32f_texture(self.src_b_tex, self._pack.src_b)
            core.update_rgba32f_texture(self.wind_tex, self._pack.wind)
        self._plan = plan_frame(self.region.boxes, mask_rects(left),
                                self.clock.substep_times(n)
                                if self.region.active else [])
        self.substeps_last_frame = n if self.region.active else 0
        self.last_update_ms = (time.perf_counter() - t0) * 1000.0

    # ---- per frame (GPU) -------------------------------------------------

    def _set(self, sh: str, name: str, *v) -> None:
        rl = self._rl
        loc = self._loc[sh][name]
        if loc < 0:
            return
        if isinstance(v[0], int) and not isinstance(v[0], bool) and len(v) == 1:
            rl.set_shader_value(self.shaders[sh], loc,
                                rl.ffi.new("int[1]", [int(v[0])]),
                                rl.ShaderUniformDataType.SHADER_UNIFORM_INT)
            return
        kind = {1: rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
                2: rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2}[len(v)]
        rl.set_shader_value(self.shaders[sh], loc,
                            rl.ffi.new(f"float[{len(v)}]",
                                       [float(x) for x in v]), kind)

    def _bind(self, sh: str, samplers) -> int:
        """Bind (uniform name, texture id) pairs to units from _FIRST_UNIT
        (call inside begin_shader_mode). Returns the count to unbind."""
        rl = self._rl
        for i, (name, tex_id) in enumerate(samplers):
            unit = _FIRST_UNIT + i
            rl.rl_active_texture_slot(unit)
            rl.rl_enable_texture(tex_id)
            self._set(sh, name, unit)
        rl.rl_active_texture_slot(0)
        return len(samplers)

    def _unbind(self, n: int) -> None:
        rl = self._rl
        for i in range(n):
            rl.rl_active_texture_slot(_FIRST_UNIT + i)
            rl.rl_disable_texture()
        rl.rl_active_texture_slot(0)

    def _stage_uniforms(self, item: PassItem, m: int) -> None:
        d = self.settings.dials
        st = item.stage
        self._set(st, "u_grid", self.w, self.h)
        self._set(st, "u_m", float(m))
        self._set(st, "u_dt", d.dt_s)
        if st == "follow_wind":
            self._set(st, "u_k_follow", d.k_follow)
        elif st == "inject":
            self._set(st, "u_t", item.t)
            self._set(st, "u_strip", d.strip_frac)
            self._set(st, "u_puff_amp", d.puff_amp)
            self._set(st, "u_puff_wobble", d.puff_wobble)
        elif st == "decay_cool":
            self._set(st, "u_lifetime", d.lifetime_s)
            self._set(st, "u_cool_rate", d.cool_rate)
            self._set(st, "u_r_eps", d.r_eps)
            self._set(st, "u_r_cut", d.r_cut)

    def run_item(self, item: PassItem) -> None:
        """Issue one planned pass (public for the GL checks)."""
        rl = self._rl
        pp = self.vel if item.field == "vel" else self.flame
        m = self.settings.m_u if item.field == "vel" else self.settings.m
        cells = [(x0 * m, y0 * m, x1 * m, y1 * m)
                 for (x0, y0, x1, y1) in item.rects]
        if item.kind == "clear":
            pp.clear_rects(cells)
            return
        sh = item.stage
        self._stage_uniforms(item, m)
        pp.begin_pass()
        rl.begin_shader_mode(self.shaders[sh])
        n = self._bind(sh, (("u_self", pp.front.id),
                            ("u_src_a", self.src_a_tex.id),
                            ("u_src_b", self.src_b_tex.id),
                            ("u_wind", self.wind_tex.id)))
        pp.draw_cell_rects(cells)
        rl.end_shader_mode()                # flushes: draws with blend off
        self._unbind(n)
        pp.end_pass()

    def simulate(self) -> int:
        """Run this frame's planned passes. Must run OUTSIDE any other
        texture mode (each pass binds its own target): the renderer calls it
        before the scene RT is begun. Returns the pass count."""
        t0 = time.perf_counter()
        plan, self._plan = self._plan, []
        for item in plan:
            self.run_item(item)
        self.passes_last_frame = len(plan)
        self.last_simulate_ms = (time.perf_counter() - t0) * 1000.0
        return len(plan)

    # ---- debug (display stage) -------------------------------------------

    def draw_debug(self, world_px_w: int, world_px_h: int) -> None:
        """The debug view over the tone-mapped picture (display space): mode
        1 the sources, mode 2 the flame R the passes made; both outline the
        active boxes. Dev only (F11 cycles, ``debug_view`` seeds)."""
        mode = self.debug_view
        if mode == 0:
            return
        rl = self._rl
        d = self.settings.dials
        if self.region.active and self._pack is not None:
            sh = "debug"
            self._set(sh, "u_grid", self.w, self.h)
            self._set(sh, "u_m", float(self.settings.m))
            self._set(sh, "u_mode", int(mode))
            self._set(sh, "u_lut_k", self.ramp.kelvin_floor,
                      self.ramp.kelvin_ceil)
            self._set(sh, "u_lut_n", float(LUT_SIZE))
            self._set(sh, "u_strip", d.strip_frac)
            self._set(sh, "u_s_vis", d.s0)
            self._set(sh, "u_r_vis", d.s0 * d.lifetime_s)
            rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA)
            rl.begin_shader_mode(self.shaders[sh])
            n = self._bind(sh, (("u_src_a", self.src_a_tex.id),
                                ("u_src_b", self.src_b_tex.id),
                                ("u_wind", self.wind_tex.id),
                                ("u_flame", self.flame.front.id),
                                ("u_lut", self.lut_tex.id)))
            rl.draw_texture_pro(
                self.src_a_tex, rl.Rectangle(0, 0, float(self.w), float(self.h)),
                rl.Rectangle(0, 0, float(world_px_w), float(world_px_h)),
                rl.Vector2(0, 0), 0.0, rl.WHITE)
            rl.end_shader_mode()
            self._unbind(n)
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
        for t in (self.src_a_tex, self.src_b_tex, self.wind_tex, self.lut_tex):
            rl.unload_texture(t)
        self.vel.unload()
        self.flame.unload()


__all__ = [
    "FireVFXSettings", "settings_from_config", "config_keys", "clamp_alpha",
    "SimClock", "cut_boxes", "boxes_mask", "mask_rects", "linger_s",
    "ActiveRegion", "read_only", "TilePack", "pack_tile_inputs",
    "blackbody_lut_kelvins", "blackbody_lut_data", "STAGES", "PassItem",
    "plan_frame", "compose_shader_source", "FireVFXPass",
    "LUT_SIZE", "MAX_BOXES", "SUBSTEP_CAP",
    "FLAG_SOLID", "FLAG_VACUUM", "FLAG_DOMAIN",
]
