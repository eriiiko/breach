"""THE swarm larvae's bodies (arc #63): every larva drawn as a lit, wriggling,
segmented grub, ALL of them in ONE instanced draw. Switched and tuned by
``[render.larvae]`` in config.toml (read every frame; a refusing door).

WHAT IT SHOWS. One larva = one instance of THE grub mesh
(``propgen.generate_larva``: the top half of a segmented tube, ~36
triangles). Its vertex shader (``shaders/larva_body.vs``) bends the template
per instance:

  * a PERISTALTIC wave (the stretch and bulge of a crawling maggot) and a
    small lateral SWAY, both travelling along the body with ONE phase per
    larva, ``2 pi (distance walked / stride_m + hash(unit_id))``. The phase
    is a function of the store's ``dist_walked`` column only: a larva that
    crawls undulates at its crawling speed; a still one (coma, blocked) is
    held in its last pose. No per-larva CPU animation, no clock.
  * a DEAD larva (``SwarmUnitDiedEvent``: the slot is already freed, so the
    body is the event's) goes LIMP over ``limp_s`` (the wriggle runs out,
    the body flattens to ``dead_flatten`` and spreads, the colour darkens to
    ``dead_rgb``), then FADES over ``fade_s`` of sim time (a paused game
    holds it). Its wriggle starts from the phase it had alive (kept from the
    last frame's pack by unit id), so it does not jump at death. The newest
    ``dead_cap`` are kept. The dot view's red-brown death FLECK is not drawn
    by this view: the limp body is the death's mark here (the fleck stays
    the dot view's, a debug picture).

HOW IT IS DRAWN. Inside the world composite's linear scene, ON the floor
under the smoke, in the dot view's slot (after the water, before the floor
gas layer), PREMULTIPLIED alpha, in ONE direct-GL ``glDrawArraysInstanced``
with an empty VAO: the template's vertices come from a small RGBA32F mesh
texture (``gl_VertexID``), each larva's four-float pair from the instance
texture (``gl_InstanceID``). Lit by the light field through THE lit-3D block
(``lit3d._COMMON_GLSL`` + ``lit3d._FIELD_SAMPLE_GLSL`` spliced into
``shaders/larva_body.fs``: the smoke's cast shadow and the light curve
included), half-Lambert + a gloss highlight + a rim, linear HDR, never
tone-mapped here. The mesh is a heightfield from above, so it needs no
depth test (larvae that overlap stack in instance order: the dead under the
living).

RENDER-ONLY. Reads the host mirror of the swarm store through
``simulation.swarm.swarm_column`` (THE read the dot view uses) -- exactly
the columns in :data:`READ_COLUMNS` -- and the death events; writes no sim
state; nothing under ``src/`` reads this module or ``[render.larvae]``. The
pure functions are testable headless; the instance data is built by numpy
in one pass (no Python loop per larva, no BLAS).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from simulation import swarm_fixed

from . import propgen
# the dot view's sRGB decode: one copy in the renderer's swarm views
from .swarm_dots import srgb_to_linear

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

# The swarm store columns this view reads (and nothing else): the test
# ``test_the_view_reads_only_its_declared_columns`` holds the pack to it.
READ_COLUMNS = ("pos_x", "pos_y", "heading", "ai_state", "valid",
                "dist_walked", "unit_id")

# ai_state codes (simulation.swarm): held equal by tests/test_larva_bodies.py
AI_COMA = 3

INST_W = 1024                  # instance texture width: larva i at
                               # (i % W, 2 (i // W) + row)
_UNIT_INST, _UNIT_MESH = 5, 6  # sampler units (above raylib's batch units)
_UNIT_LIGHT_A, _UNIT_LIGHT_B, _UNIT_SMOKE = 8, 9, 10
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_2D = 0x0DE1
_GL_TRIANGLES = 0x0004
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# [render.larvae] -- the refusing door
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LarvaSettings:
    """Every ``[render.larvae]`` dial; every key is required
    (:func:`larva_settings_from_config` refuses a missing or mistyped one)."""
    enabled: bool
    length_m: float          # body length, metres
    radius_m: float          # full lateral half-width, metres
    height_ratio: float      # body height / half-width (1 = round)
    rings: int               # geometric rings (triangles = 6 x rings)
    tail_taper: float        # profile exponent on the tail half (small = blunt)
    head_taper: float        # ... on the head half (large = pointed)
    size_jitter: float       # each larva is length x (1 +- this x hash)
    segments: int            # visible body segments (shaded grooves)
    groove_depth: float      # how dark and deep a groove reads, [0, 1]
    body_rgb: tuple          # sRGB 8-bit, the pale body
    head_rgb: tuple          # sRGB 8-bit, the darker head end
    head_start: float        # along the body (0 tail, 1 head) the head colour starts
    coma_rgb: tuple          # sRGB 8-bit, mixed in for a larva in chill coma
    coma_mix: float          # how much, [0, 1]
    dead_rgb: tuple          # sRGB 8-bit, a dead larva's colour once limp
    gloss: float             # the highlight's strength (x the local light)
    gloss_power: float       # its tightness (Blinn exponent)
    rim: float               # a rim of the local light at the silhouette
    light_z: float           # the lit-3D block's grazing key (as the props')
    peristalsis: float       # the crawl wave's stretch amplitude, [0, 0.9)
    sway: float              # the lateral wave's amplitude, share of length
    body_waves: float        # waves along the body at once
    stride_m: float          # metres walked per wriggle cycle
    limp_s: float            # a dead larva goes limp over this, s (sim clock)
    fade_s: float            # then fades out over this, s
    dead_flatten: float      # a limp body's height share, (0, 1]
    dead_cap: int            # never more dead bodies drawn at once


_KINDS = {f.name: f.type for f in fields(LarvaSettings)}
_RGB = ("body_rgb", "head_rgb", "coma_rgb", "dead_rgb")


def larva_config_keys() -> Tuple[str, ...]:
    """Every key ``[render.larvae]`` must carry."""
    return tuple(_KINDS)


def larva_settings_from_config(cfg) -> LarvaSettings:
    """``[render.larvae]`` -> settings, read every frame. A REFUSING door
    (the debris'): a missing section or key, a mistyped or out-of-range
    value raises ValueError naming it."""
    sec = getattr(getattr(cfg, "render", None), "larvae", None)
    if sec is None:
        raise ValueError("config.toml has no [render.larvae] section")
    vals = {}
    for name, kind in _KINDS.items():
        if not hasattr(sec, name):
            raise ValueError(f"[render.larvae] is missing {name!r}")
        v = getattr(sec, name)
        if kind in ("bool", bool):
            ok = isinstance(v, bool)
        elif kind in ("int", int):
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif kind in ("tuple", tuple):
            ok = (isinstance(v, (list, tuple)) and len(v) == 3
                  and all(isinstance(c, int) and not isinstance(c, bool)
                          and 0 <= c <= 255 for c in v))
            v = tuple(int(c) for c in v) if ok else v
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"[render.larvae] {name} has the wrong type: "
                             f"{v!r}")
        vals[name] = v
    s = LarvaSettings(**vals)
    for name in ("length_m", "radius_m", "height_ratio", "tail_taper",
                 "head_taper", "gloss_power", "light_z", "body_waves",
                 "stride_m", "limp_s", "fade_s"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[render.larvae] {name} must be > 0")
    for name in ("gloss", "rim", "sway"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[render.larvae] {name} must be >= 0")
    for name in ("groove_depth", "coma_mix", "head_start"):
        if not 0.0 <= getattr(s, name) <= 1.0:
            raise ValueError(f"[render.larvae] {name} must be in [0, 1]")
    if not 0.0 <= s.size_jitter < 1.0:
        raise ValueError("[render.larvae] size_jitter must be in [0, 1)")
    # the stretch is 1 + peristalsis cos(.): at 0.9 a crest nearly folds
    if not 0.0 <= s.peristalsis < 0.9:
        raise ValueError("[render.larvae] peristalsis must be in [0, 0.9)")
    if s.sway > 0.3:
        raise ValueError("[render.larvae] sway must be <= 0.3 (a fold)")
    if not 0.0 < s.dead_flatten <= 1.0:
        raise ValueError("[render.larvae] dead_flatten must be in (0, 1]")
    if s.rings < 2 or s.rings > 12:
        raise ValueError("[render.larvae] rings must be in [2, 12]")
    if s.segments < 1:
        raise ValueError("[render.larvae] segments must be >= 1")
    if s.dead_cap < 0:
        raise ValueError("[render.larvae] dead_cap must be >= 0")
    return s


# ---------------------------------------------------------------------------
# The template mesh and the per-larva instance data (pure numpy)
# ---------------------------------------------------------------------------

def mesh_key(s: LarvaSettings) -> tuple:
    """The dials the template mesh is a function of (a change re-uploads)."""
    return (s.length_m, s.radius_m, s.rings, s.height_ratio, s.tail_taper,
            s.head_taper)


def pack_mesh(s: LarvaSettings) -> np.ndarray:
    """THE grub (``propgen.generate_larva``) as the mesh texture's data:
    ``(2, n, 4)`` float32, n = its vertices; row 0 = (x, y, z metres, s
    along the body), row 1 = (normal xyz, occlusion shade)."""
    v, n, c = propgen.generate_larva(s.length_m, s.radius_m, s.rings,
                                     s.height_ratio, s.tail_taper,
                                     s.head_taper)
    out = np.zeros((2, v.shape[0], 4), np.float32)
    out[0, :, :3] = v
    # s exactly from x (the mesh's alpha carries it in 8 bits only)
    out[0, :, 3] = v[:, 0] / np.float32(s.length_m) + np.float32(0.5)
    out[1, :, :3] = n
    out[1, :, 3] = c[:, 0].astype(np.float32) / 255.0
    return out


def unit_hash(unit_id, salt: int) -> np.ndarray:
    """A render-only hash of a unit id to [0, 1) (Knuth's multiplicative
    hash, two salts): the larva's phase offset and size, stable for life."""
    u = np.asarray(unit_id, dtype=np.uint64) + np.uint64(salt)
    h = (u * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(13)
    return (h.astype(np.float64) / 4294967296.0)


def wriggle_phase(dist_walked, unit_id, stride_m: float,
                  tile_size_m: float) -> np.ndarray:
    """The wriggle phase, radians in [0, 2 pi): ``2 pi frac(walked metres /
    stride_m + hash(unit_id))``. A function of ``dist_walked`` (Q16.16
    tiles, uint32, wraps mod 2**32 = 65536 tiles: a jump once per 22 km
    walked) and the id only, so a still larva's phase never advances."""
    walked_m = (np.asarray(dist_walked, dtype=np.uint32).astype(np.float64)
                / swarm_fixed.FP_ONE) * float(tile_size_m)
    cyc = walked_m / float(stride_m) + unit_hash(unit_id, 0x9E37)
    return (TWO_PI * (cyc - np.floor(cyc))).astype(np.float32)


def larva_size(unit_id, size_jitter: float) -> np.ndarray:
    """Each larva's size factor, ``1 +- size_jitter``, stable for life."""
    return (1.0 + float(size_jitter)
            * (2.0 * unit_hash(unit_id, 0x51ED) - 1.0)).astype(np.float32)


def _in_view(x_t, y_t, view, margin: float) -> np.ndarray:
    if view is None:
        return np.ones(np.shape(x_t), bool)
    x0, y0, x1, y1 = view
    return ((x_t >= x0 - margin) & (x_t <= x1 + margin)
            & (y_t >= y0 - margin) & (y_t <= y1 + margin))


def pack_live(cols, s: LarvaSettings, px_per_tile: float,
              tile_size_m: float, view=None):
    """The living larvae of one species as instances: ``(inst, uid, phase)``
    with ``inst`` ``(2, k, 4)`` float32, k = the valid slots in ``view``
    (tiles, (x0, y0, x1, y1); None = all). Row 0 = (x, y world px, heading
    rad, wriggle phase rad); row 1 = (limp 0, alpha 1, coma 0/1, size).
    ``cols`` maps each name of :data:`READ_COLUMNS` to its column array and
    is only read. Pure numpy, one pass."""
    v = np.asarray(cols["valid"]) != 0
    x_t = swarm_fixed.dequantize(np.asarray(cols["pos_x"])[v])
    y_t = swarm_fixed.dequantize(np.asarray(cols["pos_y"])[v])
    keep = _in_view(x_t, y_t, view, s.length_m / tile_size_m)
    x_t, y_t = x_t[keep], y_t[keep]
    k = int(x_t.shape[0])
    out = np.zeros((2, k, 4), np.float32)
    uid = np.asarray(cols["unit_id"])[v][keep]
    phase = wriggle_phase(np.asarray(cols["dist_walked"])[v][keep], uid,
                          s.stride_m, tile_size_m)
    if k == 0:
        return out, uid, phase
    ppt = float(px_per_tile)
    out[0, :, 0] = x_t * ppt
    out[0, :, 1] = y_t * ppt
    out[0, :, 2] = swarm_fixed.dequantize(np.asarray(cols["heading"])[v][keep])
    out[0, :, 3] = phase
    out[1, :, 1] = 1.0
    out[1, :, 2] = np.asarray(cols["ai_state"])[v][keep] == AI_COMA
    out[1, :, 3] = larva_size(uid, s.size_jitter)
    return out, uid, phase


def dead_weights(age_s, limp_s: float, fade_s: float):
    """A dead body at ``age_s`` seconds: (limp in [0, 1], rising over
    ``limp_s``; alpha in [0, 1], 1 until limp, then falling linearly to 0
    over ``fade_s``). A negative age is fresh (0, 1)."""
    a = np.asarray(age_s, dtype=np.float64)
    limp = np.clip(a / float(limp_s), 0.0, 1.0)
    alpha = np.clip(1.0 - (a - float(limp_s)) / float(fade_s), 0.0, 1.0)
    return limp.astype(np.float32), alpha.astype(np.float32)


class DeadLarvae:
    """The limp, fading bodies of the swarm's deaths: render state only,
    driven by the death events, capped at the newest ``cap``. Positions in
    tiles, heading in radians, the birth on the SIM clock (seconds)."""

    _COLS = ("x", "y", "h", "uid", "phase", "t0")

    def __init__(self, cap: int = 2048):
        self.cap = int(cap)
        for c in self._COLS:
            setattr(self, c, np.zeros(0, np.float64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def add(self, xs_t, ys_t, headings, uids, phases, t_s: float) -> None:
        """Bodies born at ``t_s`` (event order); past the cap the OLDEST go
        first."""
        xs = np.asarray(xs_t, dtype=np.float64).reshape(-1)
        if xs.size == 0:
            return
        new = (xs, np.asarray(ys_t, np.float64).reshape(-1),
               np.asarray(headings, np.float64).reshape(-1),
               np.asarray(uids, np.float64).reshape(-1),
               np.asarray(phases, np.float64).reshape(-1),
               np.full(xs.size, float(t_s)))
        for c, a in zip(self._COLS, new):
            setattr(self, c, np.concatenate([getattr(self, c), a])[-self.cap:]
                    if self.cap > 0 else np.zeros(0, np.float64))

    def prune(self, t_s: float, life_s: float) -> None:
        keep = (float(t_s) - self.t0) < float(life_s)
        if not keep.all():
            for c in self._COLS:
                setattr(self, c, getattr(self, c)[keep])

    def set_cap(self, cap: int) -> None:
        cap = int(cap)
        if cap != self.cap:
            self.cap = cap
            for c in self._COLS:
                a = getattr(self, c)
                setattr(self, c, a[-cap:] if cap > 0 else a[:0])

    def clear(self) -> None:
        self.__init__(self.cap)

    def pack(self, t_s: float, s: LarvaSettings, px_per_tile: float,
             view=None) -> np.ndarray:
        """The bodies at ``t_s`` as instances in :func:`pack_live`'s
        layout (row 1 = (limp, alpha, 0, size))."""
        self.set_cap(s.dead_cap)
        self.prune(t_s, s.limp_s + s.fade_s)
        keep = _in_view(self.x, self.y, view, 1.0)
        k = int(keep.sum())
        out = np.zeros((2, k, 4), np.float32)
        if k == 0:
            return out
        ppt = float(px_per_tile)
        out[0, :, 0] = self.x[keep] * ppt
        out[0, :, 1] = self.y[keep] * ppt
        out[0, :, 2] = self.h[keep]
        out[0, :, 3] = self.phase[keep]
        limp, alpha = dead_weights(float(t_s) - self.t0[keep], s.limp_s,
                                   s.fade_s)
        out[1, :, 0] = limp
        out[1, :, 1] = alpha
        out[1, :, 3] = larva_size(self.uid[keep].astype(np.int64),
                                  s.size_jitter)
        return out


def lookup_phase(uids, last_uid, last_phase, fallback) -> np.ndarray:
    """The phase each of ``uids`` had in the last live pack (``last_uid``
    sorted with ``last_phase``), ``fallback`` where it was not there."""
    uids = np.asarray(uids, dtype=np.int64)
    out = np.asarray(fallback, dtype=np.float64).copy()
    if last_uid.size == 0 or uids.size == 0:
        return out
    i = np.clip(np.searchsorted(last_uid, uids), 0, last_uid.size - 1)
    hit = last_uid[i] == uids
    out[hit] = last_phase[i[hit]]
    return out


class LarvaBodies:
    """The larvae view's render state, one per renderer: this frame's
    settings and packed instances, the dead bodies, the last live phases.
    ``upload`` reads the store (only with the view on); ``on_deaths`` takes
    the death events. Pure numpy; the GL face is :class:`LarvaBodyDraw`."""

    def __init__(self):
        self.settings: Optional[LarvaSettings] = None
        self.dead = DeadLarvae()
        self.instances: Optional[np.ndarray] = None
        self._last_uid = np.zeros(0, np.int64)
        self._last_phase = np.zeros(0, np.float64)

    def refresh_settings(self, cfg) -> LarvaSettings:
        self.settings = larva_settings_from_config(cfg)
        return self.settings

    @property
    def enabled(self) -> bool:
        return self.settings is not None and self.settings.enabled

    def upload(self, gmap, t_s: float, px_per_tile: float, view=None) -> None:
        """Pack this frame's instances: the dead bodies first (drawn under),
        then every living larva of every species, from the store's host
        mirror through THE swarm column read. Off: nothing is read, the
        dead are forgotten, nothing is packed."""
        self.instances = None
        if not self.enabled:
            self.dead.clear()
            self._last_uid = np.zeros(0, np.int64)
            self._last_phase = np.zeros(0, np.float64)
            return
        from simulation import swarm as _swarm
        s = self.settings
        tile_m = float(gmap.tile_size_m)
        parts = [self.dead.pack(t_s, s, px_per_tile, view)]
        uids, phases = [], []
        if _swarm.swarm_present(gmap):
            for sp in _swarm.SWARM_SPECIES:
                cols = {c: _swarm.swarm_column(gmap, sp, c)
                        for c in READ_COLUMNS}
                inst, uid, ph = pack_live(cols, s, px_per_tile, tile_m, view)
                parts.append(inst)
                uids.append(uid)
                phases.append(ph)
        if uids:
            u = np.concatenate(uids).astype(np.int64)
            p = np.concatenate(phases).astype(np.float64)
            order = np.argsort(u, kind="stable")
            self._last_uid, self._last_phase = u[order], p[order]
        parts = [p for p in parts if p.shape[1]]
        if parts:
            self.instances = np.concatenate(parts, axis=1)

    def on_deaths(self, xs_t, ys_t, headings, uids, t_s: float) -> None:
        """Dead bodies from the death events (their PRE-ZERO position and
        heading); each starts from the phase it had alive. Off: ignored."""
        if not self.enabled:
            return
        fallback = TWO_PI * unit_hash(np.asarray(uids, np.int64), 0x9E37)
        ph = lookup_phase(uids, self._last_uid, self._last_phase, fallback)
        self.dead.add(xs_t, ys_t, headings, uids, ph, t_s)


# ---------------------------------------------------------------------------
# The GL face
# ---------------------------------------------------------------------------

_COMMON_MARKER = "// <COMMON>"
_FIELD_MARKER = "// <FIELD_SAMPLE>"


def shader_sources() -> Tuple[str, str]:
    """(vertex, fragment) source, the fragment COMPOSED with THE lit-3D
    block (``lit3d._COMMON_GLSL`` at ``// <COMMON>``,
    ``lit3d._FIELD_SAMPLE_GLSL`` at ``// <FIELD_SAMPLE>``): a larva reads
    the light field exactly as a prop and a debris chip do."""
    from .lit3d import _COMMON_GLSL, _FIELD_SAMPLE_GLSL
    fs = (SHADERS_DIR / "larva_body.fs").read_text(encoding="utf-8")
    for marker in (_COMMON_MARKER, _FIELD_MARKER):
        if marker not in fs:
            raise ValueError(f"larva_body.fs lacks '{marker}'")
    fs = fs.replace(_COMMON_MARKER, _COMMON_GLSL).replace(
        _FIELD_MARKER, _FIELD_SAMPLE_GLSL)
    return ((SHADERS_DIR / "larva_body.vs").read_text(encoding="utf-8"), fs)


def instance_texels(inst: np.ndarray, width: int = INST_W) -> np.ndarray:
    """``(2, k, 4)`` instances as the instance texture's rows: larva i at
    column ``i % width``, rows ``2 (i // width)`` (row 0) and ``+ 1``
    (row 1); ``(2 ceil(k / width), width, 4)`` float32, zero padded."""
    k = int(inst.shape[1])
    nrow = max(1, -(-k // width))
    pad = np.zeros((2, nrow * width, 4), np.float32)
    pad[:, :k] = inst
    return np.ascontiguousarray(
        pad.reshape(2, nrow, width, 4).transpose(1, 0, 2, 3)
        .reshape(2 * nrow, width, 4))


class LarvaBodyDraw:
    """Draws packed instances into the BOUND target (the scene RT, linear
    HDR), premultiplied, in ONE ``glDrawArraysInstanced``."""

    _UNIFORMS = ("u_inst", "u_mesh", "u_inst_w", "u_world_px", "u_m_to_px", "u_len_m",
                 "u_wave", "u_dead", "u_look", "u_gloss", "u_body", "u_head",
                 "u_coma", "u_dead_rgb", "texture1", "texture2", "u_light_z",
                 "u_ambient", "u_light_gain", "u_smoke_cover")

    def __init__(self, cap: int = INST_W):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        vs, fs = shader_sources()
        sh = rl.load_shader_from_memory(vs, fs)
        if sh.id == 0 or sh.id == rl.rl_get_shader_id_default():
            raise RuntimeError("larva body shader failed to compile/link")
        self.shader = sh
        self._loc = {u: rl.get_shader_location(sh, u) for u in self._UNIFORMS}
        self.inst_tex = None
        self.inst_rows = 0
        self._alloc(cap)
        self.mesh_tex = None
        self.mesh_verts = 0
        self._mesh_key = None
        self.draws_last_frame = 0
        self.instances_last_frame = 0

    def _alloc(self, cap: int) -> None:
        rows = 2 * max(1, -(-int(cap) // INST_W))
        if self.inst_tex is not None:
            self._rl.unload_texture(self.inst_tex)
        self.inst_rows = rows
        self.inst_tex = self._core.create_dynamic_rgba32f_texture(INST_W, rows)

    def _upload(self, tex, arr: np.ndarray) -> None:
        rl = self._rl
        a = np.ascontiguousarray(arr, dtype=np.float32)
        rl.update_texture_rec(tex, rl.Rectangle(0.0, 0.0, float(a.shape[1]),
                                                float(a.shape[0])),
                              rl.ffi.cast("void *",
                                          rl.ffi.from_buffer("float[]", a)))

    def _ensure_mesh(self, s: LarvaSettings) -> None:
        key = mesh_key(s)
        if key == self._mesh_key:
            return
        m = pack_mesh(s)
        n = int(m.shape[1])
        if self.mesh_tex is None or self.mesh_tex.width != n:
            if self.mesh_tex is not None:
                self._rl.unload_texture(self.mesh_tex)
            self.mesh_tex = self._core.create_dynamic_rgba32f_texture(n, 2)
        self._upload(self.mesh_tex, m)
        self.mesh_verts = n
        self._mesh_key = key

    def draw(self, inst: np.ndarray, s: LarvaSettings, *, light,
             tile_size_m: float, px_per_tile: float) -> int:
        """Draw ``inst`` ((2, k, 4) float32) lit by ``light`` (a
        ``lit3d.LightFieldCtx``). Returns the draw calls issued: 1, or 0
        when there is nothing to draw."""
        k = int(inst.shape[1])
        self.instances_last_frame = k
        self.draws_last_frame = 0
        if k == 0:
            return 0
        rl, gl = self._rl, self._gl
        from . import lit3d
        self._ensure_mesh(s)
        tex = instance_texels(inst)
        if tex.shape[0] > self.inst_rows:
            self._alloc(max(k, (self.inst_rows // 2) * INST_W * 2))
        self._upload(self.inst_tex, tex)
        # premultiplied alpha on RGB and alpha (the scene's convention)
        rl.rl_set_blend_factors_separate(rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,
                                         rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,
                                         rl.RL_FUNC_ADD, rl.RL_FUNC_ADD)
        rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM_SEPARATE)
        sh, loc = self.shader, self._loc
        U = rl.ShaderUniformDataType

        def put(name, vals):
            if loc.get(name, -1) >= 0:
                kind = {1: U.SHADER_UNIFORM_FLOAT, 2: U.SHADER_UNIFORM_VEC2,
                        3: U.SHADER_UNIFORM_VEC3,
                        4: U.SHADER_UNIFORM_VEC4}[len(vals)]
                rl.set_shader_value(sh, loc[name],
                                    rl.ffi.new(f"float[{len(vals)}]",
                                               [float(v) for v in vals]),
                                    kind)

        def put_i(name, v):
            if loc.get(name, -1) >= 0:
                rl.set_shader_value(sh, loc[name], rl.ffi.new("int *", int(v)),
                                    U.SHADER_UNIFORM_INT)

        put_i("u_inst", _UNIT_INST)
        put_i("u_mesh", _UNIT_MESH)
        put_i("u_inst_w", INST_W)
        put_i("texture1", _UNIT_LIGHT_A)
        put_i("texture2", _UNIT_LIGHT_B)
        put_i("u_smoke_cover", _UNIT_SMOKE)
        put("u_world_px", (light.world_px_w, light.world_px_h))
        put("u_m_to_px", (float(px_per_tile) / float(tile_size_m),))
        put("u_len_m", (s.length_m,))
        put("u_wave", (s.peristalsis, s.sway, s.body_waves, 0.0))
        put("u_dead", (s.dead_flatten, 0.0))
        put("u_look", (float(s.segments), s.groove_depth, s.head_start,
                       s.coma_mix))
        put("u_gloss", (s.gloss, s.gloss_power, s.rim, 0.0))
        put("u_body", tuple(srgb_to_linear(s.body_rgb)))
        put("u_head", tuple(srgb_to_linear(s.head_rgb)))
        put("u_coma", tuple(srgb_to_linear(s.coma_rgb)))
        put("u_dead_rgb", tuple(srgb_to_linear(s.dead_rgb)))
        put("u_light_z", (s.light_z,))
        put("u_ambient", tuple(light.ambient)[:3])
        put("u_light_gain", (light.light_gain,))
        cast = (light.smoke_cast if light.smoke_cover is not None
                else (0.0, 0.0, 0.0, 0.0))
        lit3d.set_smoke_shadow_uniforms(sh, light.smoke_grid, cast,
                                        light.smoke_cast2)
        lit3d.set_light_curve_uniforms(sh, light.curve)
        gl.UseProgram(sh.id)
        gl.BindVertexArray(gl.vao)
        units = [(_UNIT_INST, self.inst_tex.id), (_UNIT_MESH, self.mesh_tex.id),
                 (_UNIT_LIGHT_A, light.tex_a.id),
                 (_UNIT_LIGHT_B, light.tex_b.id)]
        if light.smoke_cover is not None:
            units.append((_UNIT_SMOKE, light.smoke_cover.id))
        for unit, tid in units:
            gl.ActiveTexture(_GL_TEXTURE0 + unit)
            gl.BindTexture(_GL_TEXTURE_2D, int(tid))
        gl.DrawArraysInstanced(_GL_TRIANGLES, 0, self.mesh_verts, k)
        for unit, _ in units:
            gl.ActiveTexture(_GL_TEXTURE0 + unit)
            gl.BindTexture(_GL_TEXTURE_2D, 0)
        gl.ActiveTexture(_GL_TEXTURE0)
        gl.BindVertexArray(0)
        gl.UseProgram(0)
        rl.end_blend_mode()
        self.draws_last_frame = 1
        return 1

    def unload(self) -> None:
        rl = self._rl
        rl.unload_shader(self.shader)
        for t in (self.inst_tex, self.mesh_tex):
            if t is not None:
                rl.unload_texture(t)
        self.inst_tex = self.mesh_tex = None


__all__ = ["READ_COLUMNS", "LarvaSettings", "larva_settings_from_config",
           "larva_config_keys", "pack_mesh", "pack_live", "wriggle_phase",
           "larva_size", "unit_hash", "dead_weights", "DeadLarvae",
           "LarvaBodies", "LarvaBodyDraw", "instance_texels", "shader_sources",
           "lookup_phase", "mesh_key", "srgb_to_linear"]
