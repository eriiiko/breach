"""Renderer core: window management, texture loading, frame setup.

Wraps pyray to provide a thin, project-specific surface. Other renderer
modules import from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pyray as rl


# ----------------------------------------------------------------------------
# Window / context
# ----------------------------------------------------------------------------

def init_window(width: int, height: int, title: str = "Breach",
                borderless: bool = False) -> None:
    """Open the application window.

    If borderless=True, set up a borderless-windowed-mode window covering the
    user's primary monitor (no decorations, fast alt-tab, taskbar may show).
    The supplied width/height are ignored in that case; the actual size comes
    from the monitor.

    Window is fixed-size in v1 — no resize handling. The borderless option
    sidesteps the resize problem by matching the monitor exactly.
    """
    if rl.is_window_ready():
        return
    flags = rl.ConfigFlags.FLAG_VSYNC_HINT
    if borderless:
        flags |= rl.ConfigFlags.FLAG_BORDERLESS_WINDOWED_MODE
    rl.set_config_flags(flags)
    rl.init_window(width, height, title)
    rl.set_target_fps(60)


def get_monitor_size() -> tuple:
    """Return (width, height) of the current monitor in pixels. Call after
    init_window so Raylib knows which monitor we're on."""
    mon = rl.get_current_monitor()
    return rl.get_monitor_width(mon), rl.get_monitor_height(mon)


def shutdown() -> None:
    if rl.is_window_ready():
        rl.close_window()


def should_close() -> bool:
    return rl.window_should_close()


# ----------------------------------------------------------------------------
# Texture management
# ----------------------------------------------------------------------------

@dataclass
class TextureSet:
    """Lazy-loaded textures for a level. Held by reference; unload on shutdown."""
    diffuse: Optional[rl.Texture] = None
    normal: Optional[rl.Texture] = None
    emissive_mask: Optional[rl.Texture] = None
    emissive_bloom: Optional[rl.Texture] = None
    background: Optional[rl.Texture] = None
    # Optional per-pixel floor heightmap (greyscale relief, 0..1). Static; used
    # by the water pass to attenuate the water alpha (raised features poke above
    # the surface). None on levels with no [art.bare] height. Sampled .r in the
    # shader, so the RGB channels are all the same on a greyscale PNG.
    height: Optional[rl.Texture] = None
    _loaded: Dict[str, rl.Texture] = field(default_factory=dict)

    def unload_all(self) -> None:
        for tex in self._loaded.values():
            rl.unload_texture(tex)
        self._loaded.clear()
        self.diffuse = None
        self.normal = None
        self.emissive_mask = None
        self.emissive_bloom = None
        self.background = None
        self.height = None


def load_texture_from_path(path: Path) -> rl.Texture:
    """Load a texture from disk, with bilinear filtering and clamping."""
    p = str(Path(path).resolve())
    tex = rl.load_texture(p)
    if tex.id == 0:
        raise RuntimeError(f"Failed to load texture: {p}")
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    return tex


def load_level_textures(level) -> TextureSet:
    """Load all art layers for a LevelData. Missing optional layers stay None."""
    ts = TextureSet()
    ts.diffuse = load_texture_from_path(level.diffuse_path)
    ts._loaded["diffuse"] = ts.diffuse
    if level.normal_path:
        ts.normal = load_texture_from_path(level.normal_path)
        ts._loaded["normal"] = ts.normal
    if level.emissive_mask_path:
        ts.emissive_mask = load_texture_from_path(level.emissive_mask_path)
        ts._loaded["emissive_mask"] = ts.emissive_mask
    if level.emissive_bloom_path:
        ts.emissive_bloom = load_texture_from_path(level.emissive_bloom_path)
        ts._loaded["emissive_bloom"] = ts.emissive_bloom
    if level.background_path:
        ts.background = load_texture_from_path(level.background_path)
        ts._loaded["background"] = ts.background
    # Optional floor heightmap (greyscale relief). Static; the water pass binds
    # it to attenuate the water alpha. Bilinear/clamp like the other art layers
    # so the shoreline reads smoothly between art texels. getattr guards stub
    # levels that predate the field.
    height_path = getattr(level, "height_path", None)
    if height_path:
        ts.height = load_texture_from_path(height_path)
        ts._loaded["height"] = ts.height
    return ts


# ----------------------------------------------------------------------------
# Dynamic textures (light field, smoke, fire — uploaded each frame)
# ----------------------------------------------------------------------------

def create_dynamic_rgba_texture(width: int, height: int) -> rl.Texture:
    """Create an RGBA texture that we'll update each frame.

    Used for the light field (50x120) and smoke/fire overlays at physics res.
    """
    blank = rl.gen_image_color(width, height, rl.BLACK)
    rl.image_format(blank, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
    tex = rl.load_texture_from_image(blank)
    rl.unload_image(blank)
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    return tex


def create_dynamic_rgba16f_texture(width: int, height: int) -> rl.Texture:
    """Create an RGBA16F (half-float) texture updated each frame.

    Used for the light field render textures (ch.05): RGB carries HDR light
    colour, alpha carries a signed light-direction component. 16-bit float is
    chosen to avoid near-dark banding at ambient ~0.01 and to carry HDR for the
    tone-map stage.
    """
    blank = rl.gen_image_color(width, height, rl.BLACK)
    rl.image_format(blank, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R16G16B16A16)
    tex = rl.load_texture_from_image(blank)
    rl.unload_image(blank)
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    return tex


def load_render_texture_rgba16f(width: int, height: int) -> rl.RenderTexture:
    """An RGBA16F (half-float) render texture with a depth renderbuffer.

    raylib's ``load_render_texture`` hard-codes RGBA8. This is the same
    construction raylib 5.5's ``LoadRenderTexture`` performs (rcore.c:
    ``rlLoadFramebuffer`` + ``rlLoadTexture`` + ``rlLoadTextureDepth`` +
    two ``rlFramebufferAttach``), through raylib's own rlgl layer, with the
    colour format swapped to ``PIXELFORMAT_UNCOMPRESSED_R16G16B16A16`` (rlgl
    maps it to GL_RGBA16F / GL_HALF_FLOAT, colour-renderable and blendable
    under GL 3.3). No raw GL: the returned struct is an ordinary raylib
    ``RenderTexture`` -- ``begin_texture_mode`` / ``unload_render_texture``
    work on it unchanged. Graphics arc G0 P1: the world's scene RT, so every
    layer can write linear HDR and ONE tone-map runs at the resolve.
    """
    fmt = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    fbo = rl.rl_load_framebuffer()
    if fbo == 0:
        raise RuntimeError("rlLoadFramebuffer failed (no GL context?)")
    rl.rl_enable_framebuffer(fbo)
    tex_id = rl.rl_load_texture(rl.ffi.NULL, int(width), int(height), fmt, 1)
    depth_id = rl.rl_load_texture_depth(int(width), int(height), True)
    rl.rl_framebuffer_attach(fbo, tex_id, rl.RL_ATTACHMENT_COLOR_CHANNEL0,
                             rl.RL_ATTACHMENT_TEXTURE2D, 0)
    rl.rl_framebuffer_attach(fbo, depth_id, rl.RL_ATTACHMENT_DEPTH,
                             rl.RL_ATTACHMENT_RENDERBUFFER, 0)
    complete = rl.rl_framebuffer_complete(fbo)
    rl.rl_disable_framebuffer()
    if not complete or tex_id == 0:
        raise RuntimeError(f"RGBA16F framebuffer {width}x{height} incomplete")
    # depth format 19 = what LoadRenderTexture records for its renderbuffer
    return rl.RenderTexture(fbo, rl.Texture(tex_id, width, height, 1, fmt),
                            rl.Texture(depth_id, width, height, 1, 19))


def update_rgba16f_texture(tex: rl.Texture, pixels_rgba: np.ndarray) -> None:
    """Upload an (H, W, 4) float16 RGBA array to an RGBA16F texture.

    Zero-copy via cffi from_buffer (same contract as update_rgba_texture):
    the half-float layout matches Raylib's PIXELFORMAT_UNCOMPRESSED_R16G16B16A16,
    so the raw bytes upload directly. Direction components may be signed; RGB
    may exceed 1.0 (HDR) — both are fine in 16F.
    """
    assert pixels_rgba.dtype == np.float16, \
        f"expected float16, got {pixels_rgba.dtype}"
    assert pixels_rgba.ndim == 3 and pixels_rgba.shape[2] == 4, \
        f"expected (H,W,4), got {pixels_rgba.shape}"
    contig = np.ascontiguousarray(pixels_rgba)
    buf = rl.ffi.from_buffer("uint16_t[]", contig)
    rl.update_texture(tex, rl.ffi.cast("void *", buf))


def create_dynamic_rgba32f_texture(width: int, height: int) -> rl.Texture:
    """Create an RGBA32F (float) texture updated each tick/frame, POINT
    filtered and clamped (data textures read with ``texelFetch``; set a
    bilinear filter yourself for a LUT). Graphics G3 F1: the flame VFX's
    per-tile source / flags / wind textures and the blackbody GPU LUT, where
    fp16's 11-bit mantissa would round a 1300 K source to the nearest 1 K and
    break the GL-vs-twin parity checks."""
    fmt = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32
    tex_id = rl.rl_load_texture(rl.ffi.NULL, int(width), int(height), fmt, 1)
    if tex_id == 0:
        raise RuntimeError(f"RGBA32F texture {width}x{height} failed")
    tex = rl.Texture(tex_id, int(width), int(height), 1, fmt)
    rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_POINT)
    rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
    return tex


def update_rgba32f_texture(tex: rl.Texture, pixels_rgba: np.ndarray) -> None:
    """Upload an (H, W, 4) float32 RGBA array to an RGBA32F texture
    (zero-copy, the :func:`update_rgba_texture` contract). Array row 0 lands
    in GL row 0 -- the first row a shader's ``texelFetch(t, ivec2(x, 0))``
    reads."""
    assert pixels_rgba.dtype == np.float32, \
        f"expected float32, got {pixels_rgba.dtype}"
    assert pixels_rgba.ndim == 3 and pixels_rgba.shape[2] == 4, \
        f"expected (H,W,4), got {pixels_rgba.shape}"
    contig = np.ascontiguousarray(pixels_rgba)
    buf = rl.ffi.from_buffer("float[]", contig)
    rl.update_texture(tex, rl.ffi.cast("void *", buf))


def read_texture_rgba32f(tex: rl.Texture) -> np.ndarray:
    """Read an RGBA32F texture back as an (H, W, 4) float32 array, GL row 0
    first (tests and debug only -- it stalls the pipeline)."""
    fmt = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32
    w, h = int(tex.width), int(tex.height)
    ptr = rl.rl_read_texture_pixels(tex.id, w, h, fmt)
    if ptr == rl.ffi.NULL:
        raise RuntimeError("rlReadTexturePixels returned NULL")
    try:
        out = np.frombuffer(rl.ffi.buffer(ptr, w * h * 16),
                            dtype=np.float32).reshape(h, w, 4).copy()
    finally:
        rl.mem_free(ptr)
    return out


class PingPong:
    """THE double-buffered render-target helper for iterative GPU passes
    (graphics arc G3 F1, docs/g3_fire_design_2026-10-23.md §9, §13.11). A
    pass reads the FRONT texture and writes the BACK target; ``end_pass``
    swaps. Never hand-roll a second FBO pair.

    Built through raylib's own rlgl layer (the ``load_render_texture_rgba16f``
    construction): RGBA32F when the FBO is complete with it (``fp32=True``,
    rlgl maps R32G32B32A32 to GL_RGBA32F), else RGBA16F (``self.fp32`` says
    which you got). Both buffers are cleared to 0 at creation. ``front`` /
    ``back`` are ordinary raylib ``Texture`` / ``RenderTexture`` structs.

    THE TRAPS (each one cost a G-arc time; the GL check covers them):

    * **Blending.** raylib's default ``BLEND_ALPHA`` would mix the shader's
      output into the old texel and corrupt simulation data (and raylib's
      ``BLEND_ADD_COLORS`` is ONE/ONE, not a replace). ``begin_pass`` turns
      blending OFF after ``begin_texture_mode`` has flushed the batch, and
      ``end_pass`` turns it back on only after ``end_texture_mode`` has
      flushed this pass's quads -- so the quads draw replace-only and no
      other draw is affected. rlgl's blend switches do NOT flush the batch
      themselves, which is why the order matters.
    * **Y flip / texel space.** Drawing into a render texture flips Y
      relative to raylib's screen coordinates. The ONE convention: shaders
      address texels with ``gl_FragCoord`` (``texelFetch(front,
      ivec2(gl_FragCoord.xy))``), never ``fragTexCoord``, and GL texel row j
      is DATA row j (for the fire VFX: cell row j from the north, the same
      orientation a numpy array uploaded with ``update_*_texture`` has).
      :meth:`draw_cell_rects` takes rects in those texel coordinates and
      does the one flip into raylib's projection. A reader outside the
      passes (a debug view, a composite) samples with ``(cell + 0.5) /
      size`` the same way -- no ``1 - v``.
    * **Depth.** No depth attachment: an iterative pass never depth-tests,
      and the renderbuffer would only cost VRAM.
    * **Precision.** fp16 pressure loses convergence; ask for ``fp32``.
    * **Sampler units.** Bind extra samplers from ``_FIRST_UNIT`` (5) up
      through rlgl, as ``GasReliefPass`` does: raylib's batch binds at most 4
      and silently drops the rest.
    * **Partial passes.** A pass that draws only some rects leaves the back
      buffer's other texels as they were TWO passes ago. The caller must keep
      everything outside its rects identical in both buffers (the fire VFX
      clears a region that leaves the domain in both, with
      :meth:`clear_rects`, and starts from zero).
    * **Reallocation.** A resolution change means a new PingPong (zeroed);
      ``unload`` the old one first.
    """

    def __init__(self, width: int, height: int, *, fp32: bool = True):
        self.width, self.height = int(width), int(height)
        self.fp32 = False
        self._rts = None
        for want32 in ((True, False) if fp32 else (False,)):
            try:
                self._rts = [self._make(want32), self._make(want32)]
                self.fp32 = want32
                break
            except RuntimeError:
                if not want32:
                    raise
        self._front = 0
        self.clear()

    def _make(self, fp32: bool) -> rl.RenderTexture:
        fmt = (rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32 if fp32
               else rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R16G16B16A16)
        w, h = self.width, self.height
        fbo = rl.rl_load_framebuffer()
        if fbo == 0:
            raise RuntimeError("rlLoadFramebuffer failed (no GL context?)")
        rl.rl_enable_framebuffer(fbo)
        tex_id = rl.rl_load_texture(rl.ffi.NULL, w, h, fmt, 1)
        rl.rl_framebuffer_attach(fbo, tex_id, rl.RL_ATTACHMENT_COLOR_CHANNEL0,
                                 rl.RL_ATTACHMENT_TEXTURE2D, 0)
        complete = rl.rl_framebuffer_complete(fbo)
        rl.rl_disable_framebuffer()
        if not complete or tex_id == 0:
            if tex_id:
                rl.rl_unload_texture(tex_id)
            rl.rl_unload_framebuffer(fbo)
            raise RuntimeError(f"PingPong {w}x{h} fp32={fp32} incomplete")
        tex = rl.Texture(tex_id, w, h, 1, fmt)
        rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_POINT)
        rl.set_texture_wrap(tex, rl.TextureWrap.TEXTURE_WRAP_CLAMP)
        return rl.RenderTexture(fbo, tex, rl.Texture(0, 0, 0, 0, 0))

    @property
    def front(self) -> rl.Texture:
        """The texture the next pass reads (the latest result)."""
        return self._rts[self._front].texture

    @property
    def back_target(self) -> rl.RenderTexture:
        return self._rts[1 - self._front]

    @property
    def vram_bytes(self) -> int:
        return 2 * self.width * self.height * (16 if self.fp32 else 8)

    def clear(self) -> None:
        """Both buffers to 0 everywhere (outside any texture mode)."""
        for rt in self._rts:
            rl.begin_texture_mode(rt)
            rl.clear_background(rl.Color(0, 0, 0, 0))
            rl.end_texture_mode()

    def begin_pass(self) -> None:
        """Bind the BACK target, blending off (see the class traps). Then:
        ``begin_shader_mode``, uniforms + samplers (the front among them),
        :meth:`draw_cell_rects`, ``end_shader_mode``, :meth:`end_pass`."""
        rl.begin_texture_mode(self.back_target)   # flushes pending draws
        rl.rl_disable_color_blend()

    def end_pass(self) -> None:
        """Flush this pass's quads (blend still off), restore blending,
        swap: the target just written becomes the front."""
        rl.end_texture_mode()
        rl.rl_enable_color_blend()
        self._front = 1 - self._front

    def draw_cell_rects(self, rects) -> None:
        """Quads over texel rects ``(x0, y0, x1, y1)`` (end-exclusive, GL
        texel rows -- the class convention), clipped to the target. The one
        place the render-texture Y flip is applied."""
        h = self.height
        for x0, y0, x1, y1 in rects:
            x0, x1 = max(0, int(x0)), min(self.width, int(x1))
            y0, y1 = max(0, int(y0)), min(h, int(y1))
            if x1 > x0 and y1 > y0:
                rl.draw_rectangle_rec(
                    rl.Rectangle(float(x0), float(h - y1), float(x1 - x0),
                                 float(y1 - y0)), rl.Color(0, 0, 0, 0))

    def clear_rects(self, rects) -> None:
        """Write 0 into ``rects`` of BOTH buffers (no read, no swap): a
        region that leaves the domain is cleared once, everywhere."""
        if not rects:
            return
        for rt in self._rts:
            rl.begin_texture_mode(rt)
            rl.rl_disable_color_blend()
            self.draw_cell_rects(rects)
            rl.end_texture_mode()
            rl.rl_enable_color_blend()

    def unload(self) -> None:
        for rt in self._rts or ():
            rl.unload_render_texture(rt)
        self._rts = None


def update_rgba_texture(tex: rl.Texture, pixels_rgba: np.ndarray) -> None:
    """Upload an (H, W, 4) uint8 RGBA array to a texture.

    Zero-copy: cffi's from_buffer reads numpy via the Python buffer protocol
    directly — no tobytes() allocation per frame. The caller's array must
    stay alive for the duration of this call (it does — pyray's update_texture
    is synchronous and copies into GPU memory before returning).
    """
    assert pixels_rgba.dtype == np.uint8, f"expected uint8, got {pixels_rgba.dtype}"
    assert pixels_rgba.ndim == 3 and pixels_rgba.shape[2] == 4, \
        f"expected (H,W,4), got {pixels_rgba.shape}"
    contig = np.ascontiguousarray(pixels_rgba)
    buf = rl.ffi.from_buffer("uint8_t[]", contig)
    rl.update_texture(tex, rl.ffi.cast("void *", buf))


# ----------------------------------------------------------------------------
# Frame lifecycle
# ----------------------------------------------------------------------------

def begin_frame(clear_color=(0, 0, 0, 255)) -> None:
    rl.begin_drawing()
    rl.clear_background(rl.Color(*clear_color))


def end_frame() -> None:
    rl.end_drawing()


def draw_fps(x: int = 10, y: int = 10) -> None:
    rl.draw_fps(x, y)


# ----------------------------------------------------------------------------
# Shader loading with fallback
# ----------------------------------------------------------------------------

def load_shader_with_fallback(vs_path: str, fs_path: str) -> rl.Shader:
    """Load a shader; if it fails to compile, return Raylib's default shader.

    Raylib doesn't expose a direct compile-error check after load_shader, but
    if the IDs are bogus the resulting shader still works in pass-through mode.
    We at least verify the files exist before calling.
    """
    if not os.path.isfile(vs_path):
        print(f"[renderer] WARN: vertex shader not found: {vs_path}, using default")
        return rl.load_shader(rl.ffi.NULL, rl.ffi.NULL)
    if not os.path.isfile(fs_path):
        print(f"[renderer] WARN: fragment shader not found: {fs_path}, using default")
        return rl.load_shader(rl.ffi.NULL, rl.ffi.NULL)
    try:
        shader = rl.load_shader(vs_path, fs_path)
        return shader
    except Exception as e:
        print(f"[renderer] WARN: shader load failed ({e}); falling back to default")
        return rl.load_shader(rl.ffi.NULL, rl.ffi.NULL)


def load_shader_from_source(vs_path: str, fs_source: str) -> rl.Shader:
    """Load a shader from a vertex-shader FILE and a fragment source held
    in memory (a file with text spliced in, e.g. ``lighting.fs`` + the smoke
    cast shadow's shared GLSL, graphics G2). Falls back to raylib's default
    shader like :func:`load_shader_with_fallback`."""
    if not os.path.isfile(vs_path):
        print(f"[renderer] WARN: vertex shader not found: {vs_path}, using default")
        return rl.load_shader(rl.ffi.NULL, rl.ffi.NULL)
    try:
        with open(vs_path, encoding="utf-8") as f:
            vs_source = f.read()
        return rl.load_shader_from_memory(vs_source, fs_source)
    except Exception as e:
        print(f"[renderer] WARN: shader load failed ({e}); falling back to default")
        return rl.load_shader(rl.ffi.NULL, rl.ffi.NULL)


__all__ = [
    "init_window", "shutdown", "should_close",
    "TextureSet", "load_texture_from_path", "load_level_textures",
    "create_dynamic_rgba_texture", "update_rgba_texture",
    "create_dynamic_rgba16f_texture", "update_rgba16f_texture",
    "load_render_texture_rgba16f",
    "create_dynamic_rgba32f_texture", "update_rgba32f_texture",
    "read_texture_rgba32f", "PingPong",
    "begin_frame", "end_frame", "draw_fps",
    "load_shader_with_fallback", "load_shader_from_source",
]
