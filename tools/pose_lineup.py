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
  (the same at posture 1.0). #184: loco_walk / loco_left / loco_right /
  loco_back / loco_diag / loco_run / loco_sprint / loco_slow (the locomotion
  at a set travel and drawn speed, on a look whose skeleton has one) and
  hit_front / hit_left / hit_back / hit_right (a directional flinch); --prop
  carbine draws a held prop on its socket. #187: ``clip=NAME@F`` holds clip
  NAME (a file clip or a ``mirror_of`` row) on every layer at fraction F of
  its length (``clip=Hit_Left_Mirror@0.4``); ``--left-handed`` draws every
  scene's unit left-handed (the mirror image, the prop on the partner hand).

The shading is a plain Lambert from one light over the albedo (no light
field, no gloss): enough to read a silhouette and the limbs, not the game's
look. Writes nothing in the repo.
"""
from __future__ import annotations

import argparse
import math
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
# #184: the locomotion (a look whose skeleton table has one): the scene name
# -> (travel relative to the facing in degrees, + = the unit's left; drawn
# speed in model m/s), driven through PoseLayer.locomote for 1.5 s; and the
# directional flinches (the hit's source relative to the facing, degrees)
LOCO_SCENES = {"loco_walk": (0.0, 1.32), "loco_left": (90.0, 1.06),
               "loco_right": (-90.0, 1.06), "loco_back": (180.0, 0.79),
               "loco_diag": (45.0, 1.32), "loco_run": (0.0, 2.6),
               "loco_sprint": (0.0, 4.25), "loco_slow": (0.0, 0.42)}
HIT_SCENES = {"hit_front": 0.0, "hit_left": 90.0, "hit_back": 180.0, "hit_right": -90.0}
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


def _run(pl: PoseLayer, look: str, scene: str, uid: int = 1, left: bool = False):
    """Drive one unit through ``scene``; returns its PoseState, or None
    when the look cannot show the scene. ``left``: a left-handed unit (#187)."""
    rig = pl.rig(look)
    bank = rig.bank
    _new = pl.new_state

    def new_state(look_, state, role=None):
        return _new(look_, state, role, left)

    def steps(ps, state, seconds, role=None):
        for _ in range(int(round(seconds / DT))):
            pl.drive(ps, state, DT, role)

    if scene == "idle":
        ps = new_state(look, "idle")
        steps(ps, "idle", 0.8)
    elif scene == "walk":
        ps = new_state(look, "walk")
        steps(ps, "walk", 1.6)
    elif scene == "aim_walk":
        ps = new_state(look, "walk")
        steps(ps, "walk", 1.0)
        for _ in range(int(0.6 / DT)):
            ps.aim_left = 1.0                    # the overlay held, no shot
            pl.drive(ps, "walk", DT)
    elif scene == "shot":
        ps = new_state(look, "idle")
        steps(ps, "idle", 0.5)
        pl.fire(ps)
        steps(ps, "idle", 0.2)
    elif scene == "flinch":
        if rig.hit_clip == NO_CLIP:
            return None
        ps = new_state(look, "idle")
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
        ps = new_state(look, "dead")
        steps(ps, "dead", 5.0)
    elif scene in ("get_up_mid", "get_up_end"):
        if not bank.has("Get_Up"):
            return None
        ps = new_state(look, "idle")
        c = bank.index("Get_Up")
        ps.clip[:] = c
        ps.time[:] = bank.duration[c] * (0.5 if scene == "get_up_mid" else 1.0)
    elif scene in LOCO_SCENES:
        if not pl.locomotes(look, "player"):
            return None
        rel, speed = LOCO_SCENES[scene]
        ps = new_state(look, "idle", "player")
        for _ in range(int(round(1.5 / DT))):
            state = pl.locomote(ps, "walk", speed * DT, math.radians(rel), True, DT,
                                "player", DT)
            pl.drive(ps, state, DT, "player")
    elif scene in HIT_SCENES:
        if rig.hit_clip == NO_CLIP:
            return None
        ps = new_state(look, "idle")
        steps(ps, "idle", 0.5)
        pl.hit(ps, 60.0, "bullet", math.radians(HIT_SCENES[scene]))
        steps(ps, "idle", 0.55)
    elif scene.startswith("clip="):               # #187: one clip, held
        name, _, frac = scene[len("clip="):].partition("@")
        if not bank.has(name):
            return None
        ps = new_state(look, "idle")
        c = bank.index(name)
        ps.clip[:] = c
        ps.time[:] = bank.duration[c] * (float(frac) if frac else 0.5)
    elif scene in ("zombie_walk", "zombie_walk_posture"):
        ps = new_state(look, "walk", "zombie")
        a = 1.0 if scene == "zombie_walk_posture" else pl.role_posture(look, "zombie")
        pl.set_posture(ps, uid, a)
        steps(ps, "walk", 1.6, "zombie")
    else:
        raise SystemExit(f"unknown scene {scene!r} (one of {SCENES})")
    return ps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--looks", default="soma_worker_male")
    ap.add_argument("--scenes", default=",".join(SCENES))  # + LOCO_SCENES, HIT_SCENES
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prop", default="",
                    help="#184: draw this [render.held_props] prop (e.g. carbine) on its "
                         "socket, on looks whose skeleton has it")
    ap.add_argument("--view-h", type=float, default=VIEW_H_M,
                    help="ortho view height, model metres (smaller = closer)")
    ap.add_argument("--side-y", type=float, default=0.95,
                    help="the side camera's target height, model metres (1.35: the hands)")
    ap.add_argument("--side-from", choices=("left", "right"), default="left",
                    help="the body side the side camera looks from (right: the carbine hand)")
    ap.add_argument("--left-handed", action="store_true",
                    help="#187: every scene's unit left-handed (the mirror image)")
    ap.add_argument("--focus-socket", action="store_true",
                    help="centre both cameras on the --prop's socket bone (a close-up of the grip)")
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
    prop = prop_model = grip = None
    if args.prop:                                 # #184: a held prop on its socket
        from renderer.held_props import grip_matrix, held_prop_settings, model_extent
        prop = held_prop_settings(CFG).props[args.prop]
        ppath = look_path(prop.model)
        prop_model = rl.load_model(str(ppath.resolve()))
        for mi in range(prop_model.materialCount):
            prop_model.materials[mi].shader = shader
        grip = grip_matrix(prop, prop.socket, model_extent(ppath))
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
                ps = _run(pl, look, scene, left=args.left_handed)
                for row, view in enumerate(("top", "side")):
                    rl.begin_texture_mode(rt)
                    rl.clear_background(rl.Color(70, 72, 80, 255))
                    if ps is not None:
                        g = pl.compose(look, [ps]).global_trs[0]
                        skinner.apply(look, model, g)
                        cx, cz = float(g[root, 0]), float(g[root, 2])
                        side_y = args.side_y
                        fb = rig.sockets.get(prop.socket, -1) if prop is not None else -1
                        fb = pl.held_bone(ps, fb)
                        if args.focus_socket and fb >= 0:
                            cx, side_y, cz = (float(v) for v in g[fb, 0:3])
                        cam = rl.Camera3D()
                        cam.projection = rl.CameraProjection.CAMERA_ORTHOGRAPHIC
                        cam.fovy = args.view_h
                        if view == "top":     # from above, the unit facing screen-down
                            cam.position = rl.Vector3(cx, 10.0, cz)
                            cam.target = rl.Vector3(cx, 0.0, cz)
                            cam.up = rl.Vector3(0.0, 0.0, -1.0)
                            ld = [0.3, -1.0, 0.4]
                        else:                 # from the body's left (+x): facing screen-left
                            sx = 10.0 if args.side_from == "left" else -10.0
                            cam.position = rl.Vector3(cx + sx, side_y, cz)
                            cam.target = rl.Vector3(cx, side_y, cz)
                            cam.up = rl.Vector3(0.0, 1.0, 0.0)
                            ld = [-0.6 if sx > 0 else 0.6, -0.5, -0.3]
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
                        bone = rig.sockets.get(prop.socket, -1) if prop is not None else -1
                        if bone >= 0:
                            from renderer.held_props import mirrored_grip, prop_world
                            pg = grip
                            if ps.mirrored:       # #187: the partner hand, the mirror grip
                                pg = mirrored_grip(pl.mirror_conj(look, bone), grip,
                                                   prop.mirror_axis)
                                bone = pl.held_bone(ps, bone)
                            m = prop_world(np.eye(4), g[bone], pg)
                            t = prop_model.transform
                            for r_ in range(4):
                                for c_ in range(4):
                                    setattr(t, f"m{4 * c_ + r_}", float(m[r_][c_]))
                            rl.draw_model_ex(prop_model, rl.Vector3(0, 0, 0),
                                             rl.Vector3(0, 1, 0), 0.0,
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
            rl.image_draw_text(sheet, f"{look}  [render.pose.skeletons.{rig.skeleton_table}]"
                               + ("  LEFT-HANDED" if args.left_handed else ""),
                               4, 4, 16, rl.WHITE)
            out = args.out / f"lineup_{look}{'_left' if args.left_handed else ''}.png"
            rl.export_image(sheet, str(out))
            rl.unload_image(sheet)
            written.append(out)
            rl.unload_model_animations(anims, n[0])
            rl.unload_model(model)
    finally:
        if prop_model is not None:
            rl.unload_model(prop_model)
        rl.unload_render_texture(rt)
        rl.unload_shader(shader)
        rl.close_window()
    for p in written:
        print(f"[pose_lineup] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
