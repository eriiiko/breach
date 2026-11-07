"""Pose lineup -- a still picture of what the pose layer composes, per look.

Render-only tool (#179). Builds the REAL pose layer from config.toml
(``[render.pose]``), drives one unit per named scene through its public API
(``new_state`` / ``drive`` / ``fire`` / ``hit`` / ``set_posture``), composes
it, puts it on the look's mesh through the game's own skinner
(``pose_skinner.CpuInjectSkinner``) and draws it in a hidden raylib window
from two orthographic cameras: TOP (the game's top-down view, the unit
facing screen-down) and SIDE (from the body's left, facing screen-left).
One PNG per look: a row per camera, a column per scene, each cell centred on
the unit's root so a clip that travels (Fall, Get_Up) stays in frame.

    python tools/pose_lineup.py --looks soma_worker_male,worker_male --out C:/tmp/soma_proof

Scenes (``--scenes``, comma list; default all):
  idle, walk, aim_walk (the fire overlay held, no shot), shot (a launch: the
  shot one-shot on the upper layer), flinch (a hit, at the flinch's
  strongest frame), fall (the dead state's clip, at its end), get_up_mid /
  get_up_end (the ``Get_Up`` clip if the look has one), zombie_walk (the
  zombie role's walk at the role's posture on the look), zombie_walk_posture
  (the same at posture 1.0).

The shading is a plain Lambert from one light over the albedo (no light
field, no gloss): enough to read a silhouette and the limbs, not the game's
look. Writes nothing in the repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pyray as rl  # noqa: E402

from config import CFG  # noqa: E402
from renderer.pose_layer import NO_CLIP, PoseLayer, pose_settings  # noqa: E402
from renderer.pose_skinner import CpuInjectSkinner  # noqa: E402
from renderer.unit_looks import look_path  # noqa: E402

SCENES = ("idle", "walk", "aim_walk", "shot", "flinch", "fall", "get_up_mid",
          "get_up_end", "zombie_walk", "zombie_walk_posture")
CELL_W, CELL_H = 240, 300
VIEW_H_M = 2.3            # ortho view height, model metres
DT = 1.0 / 60.0

_VS = """#version 330
in vec3 vertexPosition; in vec2 vertexTexCoord; in vec3 vertexNormal;
uniform mat4 mvp; uniform mat4 matModel;
out vec2 uv; out vec3 nrm;
void main() { uv = vertexTexCoord; nrm = mat3(matModel) * vertexNormal;
              gl_Position = mvp * vec4(vertexPosition, 1.0); }"""
_FS = """#version 330
in vec2 uv; in vec3 nrm; uniform sampler2D texture0; uniform vec4 colDiffuse;
uniform vec3 lightDir; out vec4 finalColor;
void main() { vec3 n = normalize(nrm); float l = max(dot(n, -lightDir), 0.0);
              vec3 c = texture(texture0, uv).rgb * colDiffuse.rgb;
              finalColor = vec4(c * (0.35 + 0.75 * l), 1.0); }"""


def _run(pl: PoseLayer, look: str, scene: str, uid: int = 1):
    """Drive one unit through ``scene``; returns its PoseState, or None
    when the look cannot show the scene."""
    rig = pl.rig(look)
    bank = rig.bank

    def steps(ps, state, seconds, role=None):
        for _ in range(int(round(seconds / DT))):
            pl.drive(ps, state, DT, role)

    if scene == "idle":
        ps = pl.new_state(look, "idle")
        steps(ps, "idle", 0.8)
    elif scene == "walk":
        ps = pl.new_state(look, "walk")
        steps(ps, "walk", 1.6)
    elif scene == "aim_walk":
        ps = pl.new_state(look, "walk")
        steps(ps, "walk", 1.0)
        for _ in range(int(0.6 / DT)):
            ps.aim_left = 1.0                    # the overlay held, no shot
            pl.drive(ps, "walk", DT)
    elif scene == "shot":
        ps = pl.new_state(look, "idle")
        steps(ps, "idle", 0.5)
        pl.fire(ps)
        steps(ps, "idle", 0.2)
    elif scene == "flinch":
        if rig.hit_clip == NO_CLIP:
            return None
        ps = pl.new_state(look, "idle")
        steps(ps, "idle", 0.5)
        pl.hit(ps, 60.0, "melee")
        # the frame where the additive moves the bones most
        best, best_ps = -1.0, None
        import copy
        for _ in range(int(3.0 / DT)):
            pl.drive(ps, "idle", DT)
            if ps.flinch_t < 0:
                break
            snap = pl.compose(look, [ps]).local_r[0]
            base_ps = copy.copy(ps)
            base_ps.add_clip = NO_CLIP
            base = pl.compose(look, [base_ps]).local_r[0]
            d = float(np.degrees(2 * np.arccos(np.clip(np.abs((snap * base).sum(-1)), 0, 1))).max())
            if d > best:
                best, best_ps = d, copy.deepcopy(ps)
        ps = best_ps
    elif scene == "fall":
        ps = pl.new_state(look, "dead")
        steps(ps, "dead", 5.0)
    elif scene in ("get_up_mid", "get_up_end"):
        if not bank.has("Get_Up"):
            return None
        ps = pl.new_state(look, "idle")
        c = bank.index("Get_Up")
        ps.clip[:] = c
        ps.time[:] = bank.duration[c] * (0.5 if scene == "get_up_mid" else 1.0)
    elif scene in ("zombie_walk", "zombie_walk_posture"):
        ps = pl.new_state(look, "walk", "zombie")
        a = 1.0 if scene == "zombie_walk_posture" else pl.role_posture(look, "zombie")
        pl.set_posture(ps, uid, a)
        steps(ps, "walk", 1.6, "zombie")
    else:
        raise SystemExit(f"unknown scene {scene!r} (one of {SCENES})")
    return ps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--looks", default="soma_worker_male")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    looks = [x for x in args.looks.split(",") if x]
    scenes = [x for x in args.scenes.split(",") if x]
    args.out.mkdir(parents=True, exist_ok=True)

    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(CELL_W, CELL_H, "pose lineup")
    shader = rl.load_shader_from_memory(_VS, _FS)
    light_loc = rl.get_shader_location(shader, "lightDir")
    pl = PoseLayer(pose_settings(CFG))
    skinner = CpuInjectSkinner(rl)
    rt = rl.load_render_texture(CELL_W, CELL_H)
    written = []
    try:
        for look in looks:
            path = str(look_path(look).resolve())
            model = rl.load_model(path)
            n = rl.ffi.new("int *", 0)
            anims = rl.load_model_animations(path, n)
            rig = pl.add_look(look, path)
            skinner.prepare(look, model, anims[0], rig.bank.skeleton.names)
            for mi in range(model.materialCount):
                model.materials[mi].shader = shader
            sheet = rl.gen_image_color(CELL_W * len(scenes), CELL_H * 2 + 24,
                                       rl.Color(40, 40, 46, 255))
            root = int(rig.bank.skeleton.roots[0])
            for col, scene in enumerate(scenes):
                ps = _run(pl, look, scene)
                for row, view in enumerate(("top", "side")):
                    rl.begin_texture_mode(rt)
                    rl.clear_background(rl.Color(70, 72, 80, 255))
                    if ps is not None:
                        g = pl.compose(look, [ps]).global_trs[0]
                        skinner.apply(look, model, g)
                        cx, cz = float(g[root, 0]), float(g[root, 2])
                        cam = rl.Camera3D()
                        cam.projection = rl.CameraProjection.CAMERA_ORTHOGRAPHIC
                        cam.fovy = VIEW_H_M
                        if view == "top":     # from above, the unit facing screen-down
                            cam.position = rl.Vector3(cx, 10.0, cz)
                            cam.target = rl.Vector3(cx, 0.0, cz)
                            cam.up = rl.Vector3(0.0, 0.0, -1.0)
                            ld = [0.3, -1.0, 0.4]
                        else:                 # from the body's left (+x): facing screen-left
                            cam.position = rl.Vector3(cx + 10.0, 0.95, cz)
                            cam.target = rl.Vector3(cx, 0.95, cz)
                            cam.up = rl.Vector3(0.0, 1.0, 0.0)
                            ld = [-0.6, -0.5, -0.3]
                        ld = np.asarray(ld) / np.linalg.norm(ld)
                        rl.set_shader_value(shader, light_loc, rl.ffi.new("float[3]", list(ld)),
                                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
                        rl.begin_mode_3d(cam)
                        if view == "side":
                            rl.draw_line_3d(rl.Vector3(cx, 0.0, cz - 1.5),
                                            rl.Vector3(cx, 0.0, cz + 1.5), rl.Color(200, 200, 90, 255))
                        else:
                            rl.draw_grid(10, 0.5)
                        rl.draw_model_ex(model, rl.Vector3(0, 0, 0), rl.Vector3(0, 1, 0), 0.0,
                                         rl.Vector3(1, 1, 1), rl.WHITE)
                        rl.end_mode_3d()
                    rl.end_texture_mode()
                    img = rl.load_image_from_texture(rt.texture)
                    rl.image_flip_vertical(img)
                    rl.image_draw(sheet, img, rl.Rectangle(0, 0, CELL_W, CELL_H),
                                  rl.Rectangle(col * CELL_W, 24 + row * CELL_H, CELL_W, CELL_H),
                                  rl.WHITE)
                    rl.unload_image(img)
                    rl.image_draw_text(sheet, f"{scene} {view}" if ps is not None else f"{scene}: n/a",
                                       col * CELL_W + 4, 24 + row * CELL_H + 4, 16, rl.WHITE)
            rl.image_draw_text(sheet, f"{look}  [render.pose.skeletons.{rig.skeleton_table}]",
                               4, 4, 16, rl.WHITE)
            out = args.out / f"lineup_{look}.png"
            rl.export_image(sheet, str(out))
            rl.unload_image(sheet)
            written.append(out)
            rl.unload_model_animations(anims, n[0])
            rl.unload_model(model)
    finally:
        rl.unload_render_texture(rt)
        rl.unload_shader(shader)
        rl.close_window()
    for p in written:
        print(f"[pose_lineup] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
