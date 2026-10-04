"""Look at a game-ready character in the REAL game, without changing the game.

Runs a Breach checkout's own `main.py` with 3D units switched on. With
`--model`, every unit (zombies too) is drawn with that one file instead of the
game's `[render.unit_looks]` table; without it, the game is shown exactly as
configured (each unit with its own look). Nothing in the checkout is edited:
the swap is one module attribute (`_MODEL_OVERRIDE`), set in this process
before `main.py` starts. The previewed model is drawn at its TRUE size (the
game scales every look by the space marine's height).

    python preview_in_game.py --root <breach checkout> [--model <file.glb>] [options] -- <main.py args>

    --model FILE          draw every unit with this model (omit: the game as configured)
    --shots PREFIX        do not open a window: run hidden through the checkout's
                          tools/e2e_drive.py and save PREFIX_<frame>.png instead
    --frames 60,150       the frames to save (with --shots)
    --focus F:ZOOM:N      from frame F, centre the camera on the Nth drawn unit
                          (or F:ZOOM:X,Y -- on tile X,Y)
                          at ZOOM px per tile (with --shots)
    --mouse X,Y           report the mouse at this screen pixel, so the cursor
                          lamp lights that spot (with --shots)

Play it:    python preview_in_game.py --root C:/Users/steen/projects/breach \\
                --model assets/models/space_marine/space_marine.glb -- --level playground
Shots of the game as configured (no --model), a wide one and the 2nd unit close:
            python preview_in_game.py --root C:/Users/steen/projects/breach \\
                --shots out/game --frames 60,150 --focus 100:40:1 -- --level playground
Use the Python the checkout's docs/dev_setup.md names for this machine.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path


def parse(argv):
    own, passthrough = argv, []
    if "--" in argv:
        split = argv.index("--")
        own, passthrough = argv[:split], argv[split + 1:]
    ap = argparse.ArgumentParser(description="Look at a game-ready character in the real game.")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=None)
    # Kept so older recorded commands still run: since #33 the game draws
    # every 3D unit untinted, so there is no tint left to remove.
    ap.add_argument("--untinted", action="store_true", help="no-op since #33")
    ap.add_argument("--shots", default="")
    ap.add_argument("--frames", default="150")
    ap.add_argument("--focus", default="")
    ap.add_argument("--mouse", default="")
    return ap.parse_args(own), passthrough


def run(argv):
    args, passthrough = parse(argv)
    model = args.model.resolve() if args.model else None
    root = args.root.resolve()
    if model is not None and not model.is_file():
        raise SystemExit("no such model file: %s" % model)
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "src"), str(root / "cpp" / "build" / "Release")]

    import pyray as rl
    import renderer.game_renderer as gr
    import renderer.unit_model_renderer as umr

    if model is not None:
        umr._MODEL_OVERRIDE = model
        print("[preview] every unit ->", model)
    else:
        print("[preview] unit looks as configured ([render.unit_looks])")

    frame = [0]
    seen = {}  # unit id -> (x, y, footprint), in first-drawn order
    shots = {int(f) for f in args.frames.split(",")} if args.shots else set()
    focus = None
    if args.focus:
        f, zoom, target = args.focus.split(":")
        if "," in target:
            tx, ty = (float(v) for v in target.split(","))
            target = (tx, ty, 0.0)
        else:
            target = int(target)
        focus = (int(f), float(zoom), target)

    init = gr.GameRenderer.__init__
    draw_units = umr.UnitModelRenderer.draw_units
    update_camera = gr.GameRenderer.update_camera
    end_drawing = rl.end_drawing

    def init_with_3d_units(self, *a, **k):
        init(self, *a, **k)
        self.cfg.use_3d_units = True
        if not self.unit_models.ready:
            self.unit_models.load()

    def draw_units_recorded(self, units, *a, **k):
        for u in units:
            if getattr(u, "alive", True):
                seen[int(getattr(u, "id", id(u)))] = (
                    float(u.x), float(u.y), float(getattr(u, "footprint", 3)))
        return draw_units(self, units, *a, **k)

    def update_camera_focused(self, dt, *a, **k):
        update_camera(self, dt, *a, **k)
        if focus and frame[0] >= focus[0] and seen:
            if isinstance(focus[2], tuple):
                x, y, fp = focus[2]
            else:
                x, y, fp = list(seen.values())[min(focus[2], len(seen) - 1)]
            self.camera.set_zoom(focus[1])
            w, h = self.camera.visible_tiles()
            self.camera.pos_tile_x = x + fp / 2.0 - w / 2.0
            self.camera.pos_tile_y = y + fp / 2.0 - h / 2.0

    def end_drawing_with_shots():
        frame[0] += 1
        if frame[0] in shots:
            image = rl.load_image_from_screen()
            path = "%s_%04d.png" % (args.shots, frame[0])
            rl.export_image(image, path)
            rl.unload_image(image)
            print("[preview] saved", path, "-- units drawn:", len(seen))
        end_drawing()

    gr.GameRenderer.__init__ = init_with_3d_units
    umr.UnitModelRenderer.draw_units = draw_units_recorded
    gr.GameRenderer.update_camera = update_camera_focused
    rl.end_drawing = end_drawing_with_shots
    if args.mouse:
        mx, my = (float(v) for v in args.mouse.split(","))
        rl.get_mouse_position = lambda: rl.Vector2(mx, my)
        rl.get_mouse_x = lambda: int(mx)
        rl.get_mouse_y = lambda: int(my)

    if args.shots:
        script = root / "tools" / "e2e_drive.py"
        sys.argv = [str(script), "--frames", str(max(shots) + 5)] + passthrough
    else:
        script = root / "main.py"
        sys.argv = [str(script)] + passthrough
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    run(sys.argv[1:])
