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
    _loaded: Dict[str, rl.Texture] = field(default_factory=dict)

    def unload_all(self) -> None:
        for tex in self._loaded.values():
            rl.unload_texture(tex)
        self._loaded.clear()
        self.diffuse = None
        self.normal = None
        self.emissive_mask = None
        self.emissive_bloom = None


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


__all__ = [
    "init_window", "shutdown", "should_close",
    "TextureSet", "load_texture_from_path", "load_level_textures",
    "create_dynamic_rgba_texture", "update_rgba_texture",
    "begin_frame", "end_frame", "draw_fps",
    "load_shader_with_fallback",
]
