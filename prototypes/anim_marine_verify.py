"""Load-verify for the Phase-0 3D marine (throwaway harness).

Bounded, headless-friendly: opens a small window, loads a rigged glTF, prints
bone/clip counts, renders TWO overlapping animated models into a top-down
orthographic Camera3D inside a RenderTexture, advances the animation, then saves
a PNG and closes. NEVER loops on window_should_close — a fixed frame budget so it
cannot hang (wrap the invocation in a shell `timeout` as a second guard).

Run:  conda run -n data python prototypes/anim_marine_verify.py [model.glb]

Verifies: import works, model is genuinely rigged (nonzero bones + >=1 clip),
animation advances, 3D composites into an RT, and depth/occlusion is correct
(the two overlapping models must occlude, proving the RT has a usable depth
buffer). Prints a PASS/FAIL line for depth so the caller can decide whether the
live module needs the world-Y painter's fallback.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pyray as rl

MODEL = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).parent.parent / "assets" / "models" / "marine" / "robot.glb")
OUT_PNG = str(Path(__file__).parent / "anim_marine_verify_out.png")

RT_W, RT_H = 512, 512
FRAMES = 45  # bounded frame budget — no infinite loop


def main() -> int:
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
    rl.init_window(RT_W, RT_H, "anim-verify")
    rl.set_target_fps(60)

    model = rl.load_model(MODEL)
    mesh_count = model.meshCount
    # raylib 6.1-dev binding: the rig lives on the mesh (boneCount / boneWeights),
    # not on Model (which exposes boneMatrices/skeleton). Take the max mesh bone
    # count as the rig size.
    bone_count = max((model.meshes[i].boneCount for i in range(mesh_count)),
                     default=0)
    print(f"[verify] model={MODEL}")
    print(f"[verify] meshCount={mesh_count} mesh.boneCount={bone_count}")

    anims_ptr = rl.ffi.new("int *", 0)
    anims = rl.load_model_animations(MODEL, anims_ptr)
    n_anims = anims_ptr[0]
    print(f"[verify] animation clips={n_anims}")
    for i in range(min(n_anims, 6)):
        a = anims[i]
        name = rl.ffi.string(a.name).decode("utf-8", "replace") if a.name else f"clip{i}"
        # 6.1-dev renamed frameCount->keyframeCount, framePoses->keyframePoses.
        print(f"[verify]   clip[{i}] name={name!r} keyframes={a.keyframeCount} bones={a.boneCount}")

    rigged = bone_count > 0 and n_anims > 0
    print(f"[verify] RIGGED={rigged}")

    # Top-down orthographic camera: eye above the scene looking straight down.
    # World up for the camera is +Z (so the model's vertical Y axis maps onto
    # the render plane). Ortho size frames a ~6-unit span.
    cam = rl.Camera3D()
    cam.position = rl.Vector3(0.0, 12.0, 0.01)   # tiny Z nudge avoids gimbal
    cam.target = rl.Vector3(0.0, 0.0, 0.0)
    cam.up = rl.Vector3(0.0, 0.0, -1.0)
    cam.fovy = 5.0                                # ortho "height" in world units
    cam.projection = rl.CameraProjection.CAMERA_ORTHOGRAPHIC

    rt = rl.load_render_texture(RT_W, RT_H)

    # Two overlapping models along the camera's view; one is closer to the
    # camera (higher Y). If depth works, the higher one occludes the lower.
    # Prefer a visibly-moving clip so the PNG shows a mid-stride pose.
    def _clip_index(want):
        for i in range(n_anims):
            nm = rl.ffi.string(anims[i].name).decode("utf-8", "replace")
            if nm == want:
                return i
        return 0
    anim = anims[_clip_index("Walk_Loop")] if n_anims else None
    frame = 0
    for f in range(FRAMES):
        if anim is not None:
            frame = (frame + 1) % max(1, anim.keyframeCount)
            rl.update_model_animation(model, anim, frame)

        rl.begin_texture_mode(rt)
        rl.clear_background(rl.Color(30, 30, 40, 255))
        rl.begin_mode_3d(cam)
        # Ground reference plane.
        rl.draw_plane(rl.Vector3(0, 0, 0), rl.Vector2(10, 10),
                      rl.Color(60, 60, 70, 255))
        # DEPTH DISCRIMINATOR: draw the NEAR model (white, raised toward the
        # top-down eye) FIRST, then the FAR model (red, on the floor) SECOND,
        # overlapping in screen-x. With a working RT depth buffer the near
        # WHITE model occludes the far RED one where they overlap. If the RT had
        # no depth buffer, painter's order would put the last-drawn RED on top —
        # an unambiguous fail signal.
        # Fully overlapping (same screen-x, same yaw): near WHITE drawn first,
        # far RED drawn second. Sampling the shared silhouette then tells us
        # which won the pixel — green-high (orange base × white) = near won =
        # depth OK; red-dominant = far (drawn last) won = painter's order only.
        _draw(model, rl.Vector3(0.0, 2.5, 0.0), 0.0, rl.WHITE)   # near, first
        _draw(model, rl.Vector3(0.0, 0.0, 0.0), 0.0, rl.Color(255, 60, 60, 255))  # far, second
        rl.end_mode_3d()
        rl.end_texture_mode()

        rl.begin_drawing()
        rl.clear_background(rl.BLACK)
        # Blit RT to screen (flip Y).
        src = rl.Rectangle(0, 0, float(RT_W), -float(RT_H))
        dst = rl.Rectangle(0, 0, float(RT_W), float(RT_H))
        rl.draw_texture_pro(rt.texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_drawing()

    # Save the RT to a PNG for eyeball inspection.
    img = rl.load_image_from_texture(rt.texture)
    rl.image_flip_vertical(img)  # RT is Y-up

    # Automated depth verdict: scan the silhouette for coloured (non-background)
    # pixels and tally near-orange (G high) vs far-red (R>>G). If near wins the
    # overlap, the RT depth buffer occludes correctly.
    near_px = far_px = 0
    for py in range(0, RT_H, 3):
        for px in range(0, RT_W, 3):
            c = rl.get_image_color(img, px, py)
            if c.r < 70 and c.g < 70:
                continue  # background
            if c.g > 90:
                near_px += 1   # orange (white-tinted base) = near model
            elif c.r > 110 and c.g < 80:
                far_px += 1    # saturated red = far model
    depth_ok = near_px > far_px
    print(f"[verify] overlap pixels: near(orange)={near_px} far(red)={far_px}")
    print(f"[verify] DEPTH_OK={depth_ok} "
          f"({'RT depth buffer occludes correctly' if depth_ok else 'PAINTERS FALLBACK NEEDED'})")

    rl.export_image(img, OUT_PNG)
    rl.unload_image(img)
    print(f"[verify] wrote {OUT_PNG}")

    rl.unload_render_texture(rt)
    rl.unload_model_animations(anims, n_anims)
    rl.unload_model(model)
    rl.close_window()
    return 0 if rigged else 2


def _draw(model, pos, yaw_deg, tint):
    # DrawModelEx: rotate about the vertical (Y) axis by yaw; scale to a marine
    # size. robot.glb is authored ~2 units tall — scale 0.9 keeps it on-plane.
    rl.draw_model_ex(model, pos, rl.Vector3(0, 1, 0), yaw_deg,
                     rl.Vector3(0.9, 0.9, 0.9), tint)


if __name__ == "__main__":
    sys.exit(main())
