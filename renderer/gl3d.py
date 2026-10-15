"""gl3d — the ONLY raw-OpenGL shim in the renderer (issue #90).

raylib (via pyray) wraps 2D textures, cubemaps and render textures but has no
3D texture, and the volumetric smoke pass (``renderer/gas_volume.py``) needs
``sampler3D`` with hardware trilinear filtering + REPEAT wrap. This module
reaches the three entry points raylib lacks through ``ctypes`` on the
process's live GL context (the one raylib/GLFW created):

  * ``glTexImage3D``   — upload the (depth, height, width, channels) volume
  * ``glBindTexture``  with ``GL_TEXTURE_3D`` on a texture unit of our choosing
  * ``glActiveTexture``

Windows-only on purpose (every Breach machine is Windows; the GL functions
newer than 1.1 come from ``wglGetProcAddress``, which needs a current
context — so call nothing here before the window exists). Anything else in
the repo that needs a GL feature raylib lacks extends THIS module rather
than opening a second raw-GL door (design §5).

Texture-unit contract: raylib's batch binds ``GL_TEXTURE_2D`` on units 0..4
(unit 0 = the draw texture, 1..4 = ``set_shader_value_texture`` slots). A
``sampler3D`` must never share a unit with a ``sampler2D`` of the same
shader (GL forbids it), so the volume lives on :data:`VOLUME_TEXTURE_UNIT`
(7), which raylib never touches.
"""
from __future__ import annotations

import ctypes
import sys
from typing import Optional

import numpy as np

# --- GL constants (gl.h) -----------------------------------------------------
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_3D = 0x806F
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_TEXTURE_WRAP_R = 0x8072
GL_LINEAR = 0x2601
GL_NEAREST = 0x2600
GL_REPEAT = 0x2901
GL_CLAMP_TO_EDGE = 0x812F
GL_UNSIGNED_BYTE = 0x1401
GL_UNPACK_ALIGNMENT = 0x0CF5
GL_RED = 0x1903
GL_RG = 0x8227
GL_RGB = 0x1907
GL_RGBA = 0x1908
GL_R8 = 0x8229
GL_RG8 = 0x822B
GL_RGB8 = 0x8051
GL_RGBA8 = 0x8058
GL_NO_ERROR = 0

# Unit the volume sampler lives on (see module header).
VOLUME_TEXTURE_UNIT = 7
# raylib's SetShaderValueTexture hands out sampler slots 1..4 only (rlgl
# MAX_BATCH_ACTIVE_TEXTURES); a FIFTH sampler silently stays on unit 0 = the
# quad's own texture (incident 2026-10-15: u_heat read the tau texture and
# painted every smoky tile red once the soot albedo rose). Extra 2D samplers
# therefore get their own units here, below the volume's.
FIELD_TEXTURE_UNIT = 6

_FORMATS = {1: (GL_R8, GL_RED), 2: (GL_RG8, GL_RG), 3: (GL_RGB8, GL_RGB),
            4: (GL_RGBA8, GL_RGBA)}


class GLUnavailable(RuntimeError):
    """Raised when the raw-GL entry points cannot be resolved (no context,
    non-Windows, or a driver without the function)."""


class _GL:
    """Lazily resolved function pointers. One instance per process."""

    def __init__(self):
        if not sys.platform.startswith("win"):
            raise GLUnavailable("gl3d is Windows-only (opengl32.dll)")
        try:
            self._lib = ctypes.WinDLL("opengl32.dll")
        except OSError as exc:            # pragma: no cover
            raise GLUnavailable(f"opengl32.dll: {exc}") from exc
        wgl = self._lib.wglGetProcAddress
        wgl.restype = ctypes.c_void_p
        wgl.argtypes = [ctypes.c_char_p]
        self._wgl = wgl
        if not self._lib.wglGetCurrentContext():
            raise GLUnavailable("no current GL context — open the window first")
        # GL 1.1 (exported directly)
        self.glGenTextures = self._direct("glGenTextures", None,
                                          [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        self.glDeleteTextures = self._direct("glDeleteTextures", None,
                                             [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        self.glBindTexture = self._direct("glBindTexture", None,
                                          [ctypes.c_uint, ctypes.c_uint])
        self.glTexParameteri = self._direct("glTexParameteri", None,
                                            [ctypes.c_uint, ctypes.c_uint, ctypes.c_int])
        self.glPixelStorei = self._direct("glPixelStorei", None,
                                          [ctypes.c_uint, ctypes.c_int])
        self.glGetError = self._direct("glGetError", ctypes.c_uint, [])
        # GL 1.2 / 1.3 (extension-style lookup)
        self.glTexImage3D = self._ext(
            "glTexImage3D", None,
            [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
             ctypes.c_uint, ctypes.c_void_p])
        self.glActiveTexture = self._ext("glActiveTexture", None, [ctypes.c_uint])

    def _direct(self, name, restype, argtypes):
        fn = getattr(self._lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def _ext(self, name, restype, argtypes):
        addr = self._wgl(name.encode("ascii"))
        if not addr:
            raise GLUnavailable(f"wglGetProcAddress({name}) returned NULL")
        proto = ctypes.WINFUNCTYPE(restype, *argtypes)
        return proto(addr)


_gl: Optional[_GL] = None


def gl() -> _GL:
    """The resolved GL entry points (resolved on first use; needs a context)."""
    global _gl
    if _gl is None:
        _gl = _GL()
    return _gl


def available() -> bool:
    """True when the shim can resolve its entry points right now."""
    try:
        gl()
        return True
    except GLUnavailable:
        return False


def create_texture_3d(volume: np.ndarray, *, wrap_xy_repeat: bool = True,
                      wrap_z_repeat: bool = False, linear: bool = True) -> int:
    """Upload a (depth, height, width, channels) uint8 array as GL_TEXTURE_3D.

    Texture coordinate ``s`` runs along ``width`` (x), ``t`` along ``height``
    (y), ``r`` along ``depth`` (the slice axis). Returns the GL texture id;
    the binding on unit 0 is restored to nothing (raylib rebinds what it
    needs on its next draw). Raises GLUnavailable / RuntimeError on failure.
    """
    g = gl()
    if volume.dtype != np.uint8 or volume.ndim != 4:
        raise ValueError(f"need (D,H,W,C) uint8, got {volume.shape} {volume.dtype}")
    depth, height, width, ch = volume.shape
    if ch not in _FORMATS:
        raise ValueError(f"channels must be 1..4, got {ch}")
    internal, fmt = _FORMATS[ch]
    data = np.ascontiguousarray(volume)
    tex = ctypes.c_uint(0)
    g.glGenTextures(1, ctypes.byref(tex))
    g.glBindTexture(GL_TEXTURE_3D, tex.value)
    g.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    filt = GL_LINEAR if linear else GL_NEAREST
    g.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filt)
    g.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filt)
    wxy = GL_REPEAT if wrap_xy_repeat else GL_CLAMP_TO_EDGE
    g.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wxy)
    g.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wxy)
    g.glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R,
                      GL_REPEAT if wrap_z_repeat else GL_CLAMP_TO_EDGE)
    g.glTexImage3D(GL_TEXTURE_3D, 0, internal, width, height, depth, 0, fmt,
                   GL_UNSIGNED_BYTE, data.ctypes.data_as(ctypes.c_void_p))
    err = g.glGetError()
    g.glBindTexture(GL_TEXTURE_3D, 0)
    if err != GL_NO_ERROR:
        g.glDeleteTextures(1, ctypes.byref(tex))
        raise RuntimeError(f"glTexImage3D failed, glGetError=0x{err:04X} "
                           f"(shape {volume.shape})")
    return int(tex.value)


def bind_texture_3d(tex_id: int, unit: int = VOLUME_TEXTURE_UNIT) -> None:
    """Bind ``tex_id`` as GL_TEXTURE_3D on texture unit ``unit`` and return
    the active unit to 0 (raylib's assumption)."""
    g = gl()
    g.glActiveTexture(GL_TEXTURE0 + int(unit))
    g.glBindTexture(GL_TEXTURE_3D, int(tex_id))
    g.glActiveTexture(GL_TEXTURE0)


def bind_texture_2d(tex_id: int, unit: int = FIELD_TEXTURE_UNIT) -> None:
    """Bind a raylib 2D texture id as GL_TEXTURE_2D on texture unit ``unit``
    (for a shader's 5th+ sampler, which raylib's own binder cannot place)."""
    g = gl()
    g.glActiveTexture(GL_TEXTURE0 + int(unit))
    g.glBindTexture(GL_TEXTURE_2D, int(tex_id))
    g.glActiveTexture(GL_TEXTURE0)


def delete_texture(tex_id: int) -> None:
    g = gl()
    tex = ctypes.c_uint(int(tex_id))
    g.glDeleteTextures(1, ctypes.byref(tex))


__all__ = ["GLUnavailable", "VOLUME_TEXTURE_UNIT", "FIELD_TEXTURE_UNIT",
           "bind_texture_2d", "available",
           "create_texture_3d", "bind_texture_3d", "delete_texture", "gl"]
