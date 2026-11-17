"""THE swarm dot view (arc #63 P4): a DIAGNOSTIC picture of the swarm units,
drawn while they have no bodies. Switched by ``[debug] swarm_dots`` in
config.toml (never a key).

WHAT IT SHOWS. One unit = one small dot (radius ``DOT_RADIUS_TILES``) with a
short tick along its heading (``TICK_LENGTH_TILES`` from its centre), so a
tile of twenty larvae reads as a crawling mass and one larva is still
findable. Colour by the unit's ``ai_state``: crawling (RUN, and EAT, which
the law runs as RUN) bone white; COMA a cold blue-grey and NO tick
(stillness is the sign). The larva law exposes no alarmed / tumbling state,
so there is no amber. A death (``SwarmUnitDiedEvent``, never the freed slot)
leaves a dark red-brown FLECK at its place that fades out over
``FLECK_FADE_S`` seconds of sim time (a paused game holds it), the newest
``FLECK_CAP`` kept.

HOW IT IS DRAWN. Inside the world composite's linear scene, ON the floor and
under the smoke (after the water, before the floor gas layer), UNLIT (an
emissive constant: a diagnostic stays visible in the dark), PREMULTIPLIED
alpha, in ONE direct-GL draw: the sprites' data goes up as one RGBA32F
texture and ``shaders/swarm_dots.vs`` builds a quad per sprite from
``gl_VertexID`` (the particle draw's idiom). The vertex data is built by
numpy in one pass (:func:`pack_dots`, :meth:`DeathFlecks.pack`): no Python
loop per unit in the frame path, no BLAS.

RENDER-ONLY. Reads the host mirror of the store (the gmap's columns, after
the resident exit download) and the death events; writes no sim state; the
sim never reads anything here. The pure functions are testable headless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# the swarm's Q16 boundary module (pure numpy): the one dequantize
from simulation import swarm_fixed

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"

# --- the look (Claude's, at Erik's licence; arc #63 P4 brief) ---------------
DOT_RADIUS_TILES = 0.12
TICK_LENGTH_TILES = 0.28
TICK_HALF_WIDTH_TILES = 0.025
FLECK_RADIUS_TILES = 0.12
FLECK_FADE_S = 1.5
FLECK_CAP = 2048

# sRGB 8-bit colours, as authored
CRAWL_SRGB = (232, 220, 184)      # bone white
COMA_SRGB = (120, 150, 185)       # cold blue-grey
FLECK_SRGB = (110, 40, 30)        # dark red-brown

# ai_state codes (simulation.swarm: AI_DEAD, AI_RUN, AI_EAT, AI_COMA); the
# coma code is held equal to the store's by tests/test_swarm_dots.py.
AI_COMA = 3

_UNIT_DATA = 5                    # sampler unit (above raylib's batch units)
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_2D = 0x0DE1
_GL_TRIANGLES = 0x0004


def srgb_to_linear(rgb) -> np.ndarray:
    """An 8-bit sRGB colour as linear RGB (the scene RT is linear HDR; the
    one resolve encodes it back), the IEC 61966-2-1 curve."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


CRAWL_LIN = srgb_to_linear(CRAWL_SRGB)
COMA_LIN = srgb_to_linear(COMA_SRGB)
FLECK_LIN = srgb_to_linear(FLECK_SRGB)


def pack_dots(pos_x_q, pos_y_q, heading_q, ai_state, valid,
              px_per_tile: float) -> np.ndarray:
    """The live units as dot sprites: ``(2, k, 4)`` float32, k = the valid
    slots. Row 0 = (x, y world px, tick dx, tick dy px: the tick's tip from
    the centre, zero in COMA); row 1 = (r, g, b linear, alpha 1). Pure numpy,
    one pass, no loop over units."""
    v = np.asarray(valid) != 0
    k = int(v.sum())
    out = np.zeros((2, k, 4), np.float32)
    if k == 0:
        return out
    ppt = float(px_per_tile)
    x = swarm_fixed.dequantize(np.asarray(pos_x_q)[v]) * ppt
    y = swarm_fixed.dequantize(np.asarray(pos_y_q)[v]) * ppt
    h = swarm_fixed.dequantize(np.asarray(heading_q)[v])     # radians
    coma = np.asarray(ai_state)[v] == AI_COMA
    tick = np.where(coma, 0.0, TICK_LENGTH_TILES * ppt)
    out[0, :, 0] = x
    out[0, :, 1] = y
    out[0, :, 2] = np.cos(h) * tick
    out[0, :, 3] = np.sin(h) * tick
    out[1, :, :3] = np.where(coma[:, None], COMA_LIN[None, :],
                             CRAWL_LIN[None, :])
    out[1, :, 3] = 1.0
    return out


def fleck_alpha(age_s) -> np.ndarray:
    """A fleck's opacity at ``age_s`` seconds: 1 at its death, falling
    linearly to 0 at ``FLECK_FADE_S`` (and 0 after; a negative age is 1)."""
    a = 1.0 - np.asarray(age_s, dtype=np.float64) / FLECK_FADE_S
    return np.clip(a, 0.0, 1.0).astype(np.float32)


class DeathFlecks:
    """The fading flecks of the swarm's deaths: render state only, driven by
    the death events, capped at the newest ``cap``. Positions in tiles, the
    birth on the SIM clock (seconds)."""

    def __init__(self, cap: int = FLECK_CAP):
        self.cap = int(cap)
        self.x = np.zeros(0, np.float64)
        self.y = np.zeros(0, np.float64)
        self.t0 = np.zeros(0, np.float64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def add(self, xs_tiles, ys_tiles, t_s: float) -> None:
        """Flecks born at ``t_s`` at the given tile positions (event order);
        past the cap the OLDEST go first."""
        xs = np.asarray(xs_tiles, dtype=np.float64).reshape(-1)
        if xs.size == 0:
            return
        ys = np.asarray(ys_tiles, dtype=np.float64).reshape(-1)
        self.x = np.concatenate([self.x, xs])[-self.cap:]
        self.y = np.concatenate([self.y, ys])[-self.cap:]
        self.t0 = np.concatenate([self.t0, np.full(xs.size, float(t_s))])[-self.cap:]

    def prune(self, t_s: float) -> None:
        """Drop the flecks that have faded out by ``t_s``."""
        keep = (float(t_s) - self.t0) < FLECK_FADE_S
        if not keep.all():
            self.x, self.y, self.t0 = self.x[keep], self.y[keep], self.t0[keep]

    def clear(self) -> None:
        self.__init__(self.cap)

    def pack(self, t_s: float, px_per_tile: float) -> np.ndarray:
        """The live flecks at ``t_s`` as sprites in :func:`pack_dots`' layout
        (no tick; row 1 = premultiplied by nothing: (r, g, b, alpha))."""
        self.prune(t_s)
        k = len(self)
        out = np.zeros((2, k, 4), np.float32)
        if k == 0:
            return out
        ppt = float(px_per_tile)
        out[0, :, 0] = self.x * ppt
        out[0, :, 1] = self.y * ppt
        out[1, :, :3] = FLECK_LIN[None, :]
        out[1, :, 3] = fleck_alpha(float(t_s) - self.t0)
        return out


def shader_sources() -> Tuple[str, str]:
    return ((SHADERS_DIR / "swarm_dots.vs").read_text(encoding="utf-8"),
            (SHADERS_DIR / "swarm_dots.fs").read_text(encoding="utf-8"))


class SwarmDotDraw:
    """The GL face: draws packed sprites into the BOUND target (the scene
    RT, linear HDR), premultiplied, in ONE direct-GL draw call."""

    _UNIFORMS = ("u_parts", "u_world_px", "u_radius_px", "u_tick_half_px")

    def __init__(self, cap: int = 1024):
        import pyray as rl
        from . import core
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        vs, fs = shader_sources()
        sh = rl.load_shader_from_memory(vs, fs)
        if sh.id == 0 or sh.id == rl.rl_get_shader_id_default():
            raise RuntimeError("swarm dot shader failed to compile/link")
        self.shader = sh
        self._loc = {u: rl.get_shader_location(sh, u) for u in self._UNIFORMS}
        self.tex = None
        self.width = 0
        self._alloc(cap)
        self.sprites_last_frame = 0

    def _alloc(self, cap: int) -> None:
        if self.tex is not None:
            self._rl.unload_texture(self.tex)
        self.width = max(1, int(cap))
        self.tex = self._core.create_dynamic_rgba32f_texture(self.width, 2)

    def draw(self, sprites: np.ndarray, *, world_px_w: float, world_px_h: float,
             px_per_tile: float) -> int:
        """Draw ``sprites`` ((2, k, 4) float32). Returns the draw calls
        issued (0 when there is nothing to draw)."""
        k = int(sprites.shape[1])
        self.sprites_last_frame = k
        if k == 0:
            return 0
        rl, gl = self._rl, self._gl
        if k > self.width:
            self._alloc(max(k, 2 * self.width))
        a = np.ascontiguousarray(sprites, dtype=np.float32)
        rl.update_texture_rec(self.tex, rl.Rectangle(0.0, 0.0, float(k), 2.0),
                              rl.ffi.cast("void *",
                                          rl.ffi.from_buffer("float[]", a)))
        # premultiplied alpha on RGB and alpha (the scene's convention: the
        # floor smoke, the water, the debris chips)
        rl.rl_set_blend_factors_separate(rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,
                                         rl.RL_ONE, rl.RL_ONE_MINUS_SRC_ALPHA,
                                         rl.RL_FUNC_ADD, rl.RL_FUNC_ADD)
        rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM_SEPARATE)
        gl.UseProgram(self.shader.id)
        loc = self._loc
        ppt = float(px_per_tile)
        if loc["u_parts"] >= 0:
            gl.Uniform1i(loc["u_parts"], _UNIT_DATA)
        if loc["u_world_px"] >= 0:
            gl.Uniform2f(loc["u_world_px"], float(world_px_w), float(world_px_h))
        if loc["u_radius_px"] >= 0:
            gl.Uniform1f(loc["u_radius_px"], DOT_RADIUS_TILES * ppt)
        if loc["u_tick_half_px"] >= 0:
            # never thinner than ~a pixel, or a far zoom loses the heading
            gl.Uniform1f(loc["u_tick_half_px"],
                         max(TICK_HALF_WIDTH_TILES * ppt, 0.6))
        gl.BindVertexArray(gl.vao)
        gl.ActiveTexture(_GL_TEXTURE0 + _UNIT_DATA)
        gl.BindTexture(_GL_TEXTURE_2D, int(self.tex.id))
        gl.DrawArrays(_GL_TRIANGLES, 0, 6 * k)
        gl.ActiveTexture(_GL_TEXTURE0 + _UNIT_DATA)
        gl.BindTexture(_GL_TEXTURE_2D, 0)
        gl.ActiveTexture(_GL_TEXTURE0)
        gl.BindVertexArray(0)
        gl.UseProgram(0)
        rl.end_blend_mode()
        return 1

    def unload(self) -> None:
        self._rl.unload_shader(self.shader)
        if self.tex is not None:
            self._rl.unload_texture(self.tex)
            self.tex = None


__all__ = ["DOT_RADIUS_TILES", "TICK_LENGTH_TILES", "FLECK_FADE_S",
           "FLECK_CAP", "CRAWL_LIN", "COMA_LIN", "FLECK_LIN", "pack_dots",
           "fleck_alpha", "DeathFlecks", "SwarmDotDraw", "srgb_to_linear"]
