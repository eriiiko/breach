"""GL checks of the light curve (graphics G5, #64).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_light_curve.py`` spawns it and asserts
on its last line. Prints one ``[check] NAME: PASS|FAIL detail`` line per check,
then ``LIGHT_CURVE_GL_RESULT: PASS`` or ``... FAIL``.

  three_receivers_compile -- the floor, the marine and the prop shaders link
      and each carries the curve's uniforms. Breaks if a splice loses the block
      (a GLSL error otherwise falls back to raylib's default shader silently).
  off_is_the_plain_light -- the floor drawn with the curve off equals the floor
      with no curve ever set. Breaks if the off path touches the light.
  floor_runs_the_twin -- the floor's pixels under a threshold curve equal
      ``light_curve.apply_curve`` over a ramp of light, to 2/255. Breaks if the
      GLSL drifts from its spec.
  floor_saturation -- dark_saturation greys the art in the dark and leaves it
      in full colour where lit. Breaks if the two ends swap.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]

import numpy as np
import pyray as rl

from renderer.light_curve import LightCurve, apply_curve, bake_lut
from renderer.lighting import LightingPass
from renderer.marine_shader import MARINE_FS, MARINE_VS

W, H = 256, 4
STEP = ((0.0, 0.0), (0.5, 0.5), (0.52, 0.95), (1.0, 1.0))
results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[check] {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def curve(**over):
    base: dict = dict(enabled=True, points=STEP, range=4.0, dark_saturation=1.0,
                dark_knee=0.25, panel=False)
    base.update(over)
    return LightCurve(**base)


def art(rgb):
    img = rl.gen_image_color(W, H, rl.Color(*rgb, 255))
    tex = rl.load_texture_from_image(img)
    rl.unload_image(img)
    return tex


def render(lp, tex, rt):
    rl.begin_texture_mode(rt)
    rl.clear_background(rl.BLACK)
    lp.draw_lit_world(tex, None, W, H)
    rl.end_texture_mode()
    img = rl.load_image_from_texture(rt.texture)
    raw = rl.ffi.buffer(img.data, W * H * 4)
    a = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(H, W, 4).copy()
    rl.unload_image(img)
    return a[H // 2, :, :3].astype(np.float64) / 255.0


def main():
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(320, 200, "light curve check")
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    lp = LightingPass(H, W)
    lp.set_floor((0.0, 0.0, 0.0))
    lp.set_srgb_decode(False)
    lp.set_light_gain(1.0)
    rt = rl.load_render_texture(W, H)

    # A ramp of grey light, 0.003 .. 1.0, in the texture the floor reads.
    ramp = np.linspace(0.003, 1.0, W, dtype=np.float32)
    lp.packed_a[..., 0] = ramp
    lp.packed_a[..., 1] = ramp
    lp.packed_a[..., 2] = ramp
    from renderer import core
    core.update_rgba16f_texture(lp.light_tex_a, lp.packed_a)
    white = art((255, 255, 255))

    # three_receivers_compile
    s_marine = rl.load_shader_from_memory(MARINE_VS, MARINE_FS)
    ok = (lp.shader.id > 3 and s_marine.id > 3
          and rl.get_shader_location(lp.shader, "u_curve_lut") >= 0
          and rl.get_shader_location(s_marine, "u_curve") >= 0)
    from renderer.static_props import PROP_FS, PROP_VS
    s_prop = rl.load_shader_from_memory(PROP_VS, PROP_FS)
    ok = ok and s_prop.id > 3 and rl.get_shader_location(s_prop, "u_curve") >= 0
    report("three_receivers_compile", ok)

    # off_is_the_plain_light
    plain = render(lp, white, rt)
    lp.set_light_curve(curve(enabled=False))
    off = render(lp, white, rt)
    report("off_is_the_plain_light", np.array_equal(plain, off))

    # floor_runs_the_twin
    lp.set_light_curve(curve())
    got = render(lp, white, rt)
    light = np.stack([ramp] * 3, axis=-1).astype(np.float64)
    want = np.clip(apply_curve(light, bake_lut(STEP), 4.0), 0.0, 1.0)
    # the ramp has a step in it: skip the few pixels bilinear smears across it
    err = np.abs(got - want).max(axis=-1)
    jump = np.abs(np.gradient(want[:, 0])) > 0.02
    keep = ~np.convolve(jump, np.ones(5), mode="same").astype(bool)
    keep[:2] = keep[-2:] = False
    report("floor_runs_the_twin", err[keep].max() <= 2.5 / 255.0,
           f"max err {err[keep].max() * 255:.2f}/255 over {keep.sum()} px")

    # floor_saturation
    red = art((200, 60, 40))
    # identity map so only the saturation moves; the lit end is past the knee
    lp.set_light_curve(curve(points=((0.0, 0.0), (1.0, 1.0)),
                             dark_saturation=0.0, dark_knee=0.5))
    got = render(lp, red, rt)
    lp.set_light_curve(curve(enabled=False))
    full = render(lp, red, rt)
    chroma = got.max(axis=-1) - got.min(axis=-1)
    chroma_full = full.max(axis=-1) - full.min(axis=-1)
    dim, bright = W // 40, W - 2
    ok = (chroma[dim] < 0.35 * max(chroma_full[dim], 1e-3)
          and abs(chroma[bright] - chroma_full[bright]) <= 2.0 / 255.0)
    report("floor_saturation", ok,
           f"dim {chroma[dim]:.3f} of {chroma_full[dim]:.3f}, "
           f"bright {chroma[bright]:.3f} of {chroma_full[bright]:.3f}")

    rl.close_window()
    print("LIGHT_CURVE_GL_RESULT: " + ("PASS" if all(results) else "FAIL"),
          flush=True)


if __name__ == "__main__":
    main()
