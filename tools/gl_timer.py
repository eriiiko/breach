"""tools/gl_timer.py -- GPU time of a block of GL work, measured on the GPU.

The bench instrument behind ``tools/lighting_demo.py --fire-vfx-perf``
(graphics arc G3 F1; the method G1/G2 used ad hoc, docs/graphics_direction
§11-§12): a ``GL_TIME_ELAPSED`` query around the block, read back after the
frame. CPU time (``time.perf_counter``) and GPU time are reported SEPARATELY:
rlgl draw calls are asynchronous, so the CPU span of a pass chain says what
the Python + driver submission costs and nothing about the shaders.

raylib exposes no query objects, so this reaches the GL entry points through
the current context with ctypes (``wglGetProcAddress``; Windows only, like the
rest of the toolchain). Dev-only: never imported by the game.

    t = GLTimer()
    t.begin(); ...GL work...; t.end()
    ms = t.result_ms()          # blocks until the GPU has finished the block

``begin``/``end`` flush raylib's batch first, so the block's own draws land
inside the query and earlier ones do not.
"""
from __future__ import annotations

import ctypes

GL_TIME_ELAPSED = 0x88BF
GL_QUERY_RESULT = 0x8866


class _GL:
    _inst = None

    def __init__(self):
        gl = ctypes.WinDLL("opengl32")
        get = gl.wglGetProcAddress
        get.restype = ctypes.c_void_p
        get.argtypes = [ctypes.c_char_p]

        def fn(name, *argtypes):
            addr = get(name.encode("ascii"))
            if not addr:
                raise RuntimeError(f"{name} unavailable (no GL context?)")
            return ctypes.WINFUNCTYPE(None, *argtypes)(addr)

        u32 = ctypes.c_uint
        self.gen = fn("glGenQueries", ctypes.c_int, ctypes.POINTER(u32))
        self.delete = fn("glDeleteQueries", ctypes.c_int, ctypes.POINTER(u32))
        self.begin = fn("glBeginQuery", u32, u32)
        self.end = fn("glEndQuery", u32)
        self.get_u64 = fn("glGetQueryObjectui64v", u32, u32,
                          ctypes.POINTER(ctypes.c_uint64))

    @classmethod
    def get(cls) -> "_GL":
        if cls._inst is None:
            cls._inst = _GL()
        return cls._inst


class GLTimer:
    """One reusable ``GL_TIME_ELAPSED`` query."""

    def __init__(self):
        self._gl = _GL.get()
        q = ctypes.c_uint(0)
        self._gl.gen(1, ctypes.byref(q))
        self._q = q
        self._open = False
        self._pending = False

    def begin(self) -> None:
        import pyray as rl
        rl.rl_draw_render_batch_active()
        self._gl.begin(GL_TIME_ELAPSED, self._q.value)
        self._open = True

    def end(self) -> None:
        import pyray as rl
        rl.rl_draw_render_batch_active()
        self._gl.end(GL_TIME_ELAPSED)
        self._open = False
        self._pending = True

    def result_ms(self) -> float:
        """The block's GPU time in ms (blocks until it is available)."""
        if not self._pending:
            return 0.0
        v = ctypes.c_uint64(0)
        self._gl.get_u64(self._q.value, GL_QUERY_RESULT, ctypes.byref(v))
        self._pending = False
        return v.value / 1.0e6

    def delete(self) -> None:
        self._gl.delete(1, ctypes.byref(self._q))


__all__ = ["GLTimer"]
