"""SPIKE — one marine on one lit tile, in a real WorldComposite + LightingPass.

De-risks the P1 unknowns from the design (v2 §Spike) BEFORE folding into the
module:
  (a) matModel/matNormal upload to a custom material shader (already confirmed
      by introspect_model.py: loc slots auto-populate; DrawMesh uploads them).
  (b) world_uv (worldPos.xz / world_px) lands on the right tile with the ship's
      Y-orientation: a marine near the TOP (grid y small) sits in the lit lamp
      region; a marine near the BOTTOM sits in darkness — matching the flat lit
      "ship" floor drawn beside it. Proves no v-flip is needed.
  (c) light textures bind via MATERIAL_MAP slots (METALNESS/NORMAL -> texture1/2),
      NOT pre-bound uniforms.
  (d) the shader is set on materials[1] AND materials[2] and declares no
      boneMatrices (no double transform on the CPU-skinned mesh).

Saves scratchpad/spike_lit_marine.png. Run:
  conda run -n data python scratchpad/spike_lit_marine.py
"""
import sys
from pathlib import Path

import numpy as np
import pyray as rl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import breach_physics as bp
from renderer import core
from renderer.world_composite import WorldComposite
from renderer.lighting import LightingPass
from renderer.unit_model_renderer import UnitModelRenderer
from renderer.marine_shader import load_marine_shader

GRID_W, GRID_H = 40, 48
WPT = 48.0
AMBIENT = (0.10, 0.10, 0.13)
GAIN = 1.0


class FakeUnit:
    def __init__(self, uid, x, y, facing=1.5708, footprint=3):
        self.id = uid
        self.x = x
        self.y = y
        self.facing = facing
        self.footprint = footprint
        self.alive = True
        self.move_path = None


def fill_light_field(lp):
    """Hand-build a diagnostic field: one warm lamp in the UPPER area, radial
    1/r falloff, direction pointing toward the lamp. Bright at top, dark at
    bottom — so the Y-orientation is unambiguous in the saved PNG."""
    lp.light_rgb.fill(0.0)
    lp.light_dx.fill(0.0)
    lp.light_dy.fill(0.0)
    lamp_x, lamp_y = GRID_W * 0.35, GRID_H * 0.22
    color = np.array([1.0, 0.35, 0.22], dtype=np.float32)  # warm red-orange
    intensity = 6.0
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            dx = lamp_x - (gx + 0.5)
            dy = lamp_y - (gy + 0.5)
            dist = float(np.hypot(dx, dy))
            atten = intensity / (1.0 + dist * dist * 0.06)
            lp.light_rgb[gy, gx] = color * atten
            if dist > 1e-4:
                lp.light_dx[gy, gx] = dx / dist
                lp.light_dy[gy, gx] = dy / dist
    lp.light_map[:] = lp.light_rgb.max(axis=2)
    # Pack + upload exactly like LightingPass.compute_light_field.
    lp.packed_a[..., 0:3] = lp.light_rgb
    lp.packed_a[..., 3] = lp.light_dx
    lp.packed_b[..., 0:3] = 0.0
    lp.packed_b[..., 3] = lp.light_dy
    core.update_rgba16f_texture(lp.light_tex_a, lp.packed_a)
    core.update_rgba16f_texture(lp.light_tex_b, lp.packed_b)


def main():
    core.init_window(900, 700, title="spike lit marine")

    world = WorldComposite(GRID_W, GRID_H, WPT)
    raycaster = bp.Raycaster()
    lp = LightingPass(raycaster, GRID_H, GRID_W)
    lp.set_ambient(AMBIENT)
    lp.set_light_gain(GAIN)
    lp.set_srgb_decode(True)
    lp.set_use_normal(False)   # flat grey "ship" = pure field visualization
    fill_light_field(lp)

    # Flat mid-grey diffuse "ship floor" so the field reads directly.
    img = rl.gen_image_color(GRID_W, GRID_H, rl.Color(140, 140, 140, 255))
    diffuse = rl.load_texture_from_image(img)
    rl.unload_image(img)
    rl.set_texture_filter(diffuse, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    umr = UnitModelRenderer()
    umr.load()
    assert umr.ready, "model failed to load"
    model = umr.model

    # --- SPIKE wiring (the mechanics P1 will fold in) ---
    ms = load_marine_shader()
    print("marine shader id =", ms.shader.id)
    MM = rl.MaterialMapIndex
    for mi in (1, 2):   # mesh materials; material 0 is dead
        model.materials[mi].shader = ms.shader
        model.materials[mi].maps[MM.MATERIAL_MAP_METALNESS].texture = lp.light_tex_a
        model.materials[mi].maps[MM.MATERIAL_MAP_NORMAL].texture = lp.light_tex_b
    ms.set_frame_uniforms(AMBIENT, GAIN, world.world_px_w, world.world_px_h,
                          normal_y_sign=1.0, view_dir=(0.0, 1.0, 0.0))

    cam = UnitModelRenderer.make_camera(world.world_px_w, world.world_px_h)
    scale = (6.0 * WPT) / umr._native_height
    idle = umr._anims[umr._clip_index.get("Idle_Loop", 0)]

    # Marines: one in the bright lamp region (top), one in the dark bottom, one
    # rotated 90 deg beside the lit one (eyeball that the normal rotates).
    lit = FakeUnit(1, GRID_W * 0.30 - 1.5, GRID_H * 0.22 - 1.5)
    lit_yaw = FakeUnit(2, GRID_W * 0.30 + 4.5, GRID_H * 0.22 - 1.5, facing=0.0)
    dark = FakeUnit(3, GRID_W * 0.70, GRID_H * 0.82)

    def draw_marine(u, yaw_deg):
        rl.update_model_animation(model, idle, 0)
        cx = (float(u.x) + u.footprint / 2.0) * WPT
        cy = (float(u.y) + u.footprint / 2.0) * WPT
        rl.draw_model_ex(model, rl.Vector3(cx, 0.0, cy),
                         rl.Vector3(0.0, 1.0, 0.0), yaw_deg,
                         rl.Vector3(scale, scale, scale),
                         rl.Color(90, 200, 90, 255))

    # Compose into the world RT.
    world.begin(clear_color=(0, 0, 0, 0))
    lp.draw_lit_world(diffuse, None, world.world_px_w, world.world_px_h)
    rl.begin_mode_3d(cam)
    draw_marine(lit, 90.0)
    draw_marine(lit_yaw, 0.0)
    draw_marine(dark, 90.0)
    rl.end_mode_3d()
    world.end()

    # Export the world RT (Y-up) flipped to a PNG.
    rt_img = rl.load_image_from_texture(world.rt.texture)
    rl.image_flip_vertical(rt_img)
    out = str(ROOT / "scratchpad" / "spike_lit_marine.png")
    rl.export_image(rt_img, out)
    rl.unload_image(rt_img)
    print("saved", out)

    # Also drop a couple of numeric probes: the field value under each marine's
    # foot tile, so the PNG reading is anchored to data.
    for name, u in [("lit", lit), ("dark", dark)]:
        gx = int(u.x + u.footprint // 2)
        gy = int(u.y + u.footprint // 2)
        print(f"  {name} foot tile ({gx},{gy}) incoming_rgb="
              f"{lp.light_rgb[gy, gx]}")

    umr.unload()
    rl.unload_texture(diffuse)
    rl.unload_shader(ms.shader)
    world.unload()
    core.shutdown()
    print("OK")


if __name__ == "__main__":
    main()
