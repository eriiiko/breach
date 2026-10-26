"""Tool-only procedural arm client of Breach's existing skin/render seam.

--root is mandatory: this main-based art worktree predates the renderer helpers.
No files in that checkout are changed. The renderer gets a process-local raylib
proxy; the installed pyray package is untouched. Only this model's pose argument
is replaced, and the real update_model_animation performs skinning/buffer upload.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import runpy
import subprocess
import sys
import time

import numpy as np
import arm_math as arm

HOME = Path(__file__).resolve().parent.parent
MODEL = HOME / "prototype" / "octopus_arm.glb"


class LiveArm:
    def __init__(self, rl, model):
        self.rl, self.model = rl, model
        self.pose_array = rl.ffi.new("Transform[]", model.boneCount)
        self.frames = rl.ffi.new("Transform*[]", [self.pose_array])
        self.animation = rl.ffi.new("ModelAnimation *")
        self.animation.boneCount = model.boneCount
        self.animation.bones = model.bones
        self.animation.framePoses = self.frames
        for field in ("frameCount", "keyframeCount"):
            if hasattr(self.animation, field):
                setattr(self.animation, field, 1)
                break
        self.indices = {}
        self.bind_rotations = {}
        self.names = []
        for i in range(model.boneCount):
            name = rl.ffi.string(model.bones[i].name).decode()
            self.names.append(name)
            self.pose_array[i] = model.bindPose[i]
            if name.startswith("arm_"):
                self.indices[i] = int(name[4:])
                q = model.bindPose[i].rotation
                self.bind_rotations[i] = (q.x, q.y, q.z, q.w)
        if set(self.indices.values()) != set(range(arm.SEGMENTS + 1)):
            raise ValueError(f"wrong arm rig: {self.names}")

    def write_pose(self, points, rotations):
        for bone, joint in self.indices.items():
            tr = self.pose_array[bone]
            tr.translation.x, tr.translation.y, tr.translation.z = points[joint].tolist()
            x, y, z, w = arm.quaternion(rotations[joint])
            bx, by, bz, bw = self.bind_rotations[bone]
            tr.rotation.x = w * bx + x * bw + y * bz - z * by
            tr.rotation.y = w * by - x * bz + y * bw + z * bx
            tr.rotation.z = w * bz + x * by - y * bx + z * bw
            tr.rotation.w = w * bw - x * bx - y * by - z * bz

    def update(self, curl, contact):
        self.write_pose(*arm.pose(curl, contact))
        self.rl.update_model_animation(self.model, self.animation[0], 0)


def benchmark(rl, args):
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(128, 128, "Arm skin benchmark (hidden)")
    model = rl.load_model(str(MODEL))
    driver = LiveArm(rl, model)
    data = arm.geometry()
    lookup = {tuple(np.round(v, 5)): i for i, v in enumerate(data["vertices"])}
    maps, max_bind_error = [], 0.
    for i in range(model.meshCount):
        mesh = model.meshes[i]
        vertices = np.frombuffer(rl.ffi.buffer(mesh.vertices, mesh.vertexCount * 12), dtype=np.float32).reshape(-1, 3)
        indices = []
        for v in vertices:
            match = lookup.get(tuple(np.round(v, 5)))
            if match is None:
                match = int(np.argmin(np.sum((data["vertices"] - v) ** 2, axis=1)))
            indices.append(match)
        indices = np.array(indices)
        distances = np.linalg.norm(data["vertices"][indices] - vertices, axis=1)
        max_bind_error = max(max_bind_error, float(distances.max()))
        maps.append(indices)
    errors = []
    for t in np.linspace(0, arm.PERIOD, 49):
        controls = arm.cycle_weights(t)
        driver.update(*controls)
        expected = arm.skin(data["vertices"], data["joints"], data["weights"], *arm.pose(*controls))
        for i, indices in enumerate(maps):
            mesh = model.meshes[i]
            actual = np.frombuffer(rl.ffi.buffer(mesh.animVertices, mesh.vertexCount * 12), dtype=np.float32).reshape(-1, 3)
            errors.append(float(np.linalg.norm(actual - expected[indices], axis=1).max()))
    assert max_bind_error < 2e-6, max_bind_error
    assert max(errors) < 1e-5, max(errors)
    for _ in range(20):
        driver.update(.4, .3)
    batches = {}
    for count in (1, 8):
        samples = []
        for sample in range(args.samples):
            evaluate, write, skin = 0., 0., 0.
            start = time.perf_counter_ns()
            for i in range(count):
                controls = arm.cycle_weights(sample * .017 + i * .37)
                t0 = time.perf_counter_ns()
                points, rotations = arm.pose(*controls)
                t1 = time.perf_counter_ns()
                driver.write_pose(points, rotations)
                t2 = time.perf_counter_ns()
                rl.update_model_animation(model, driver.animation[0], 0)
                t3 = time.perf_counter_ns()
                evaluate += (t1 - t0) / 1e6
                write += (t2 - t1) / 1e6
                skin += (t3 - t2) / 1e6
            samples.append([evaluate, write, skin, (time.perf_counter_ns() - start) / 1e6])
        values = np.array(samples)
        batches[str(count)] = {name: dict(median_ms=float(np.median(values[:, i])),
                                            p95_ms=float(np.percentile(values[:, i], 95)))
                               for i, name in enumerate(("curve", "write_joint_transforms", "raylib_skin_and_upload", "total"))}
    report = dict(root=str(args.root), root_revision=subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
                  python=sys.version, platform=platform.platform(), processor=platform.processor(),
                  model_meshes=model.meshCount, model_joints=model.boneCount,
                  loaded_vertices=sum(model.meshes[i].vertexCount for i in range(model.meshCount)),
                  loaded_triangles=sum(model.meshes[i].triangleCount for i in range(model.meshCount)),
                  bind_vertex_max_error_m=max_bind_error, raylib_vs_numpy_max_error_m=max(errors),
                  parity_poses=49, samples_per_batch=args.samples, timings=batches,
                  methodology="perf_counter_ns wall time, 20 warmups; sequential independent arm poses on one shared model, matching UnitModelRenderer's shared-model seam. Skin call includes driver buffer submission, not GPU completion. Excludes drawing, shading, shadows, game simulation. CPU timings are not whole-game FPS.")
    out = HOME / "prototype" / "runtime_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    rl.unload_model(model)
    rl.close_window()


def main():
    own, passthrough = sys.argv[1:], []
    if "--" in own:
        split = own.index("--")
        own, passthrough = own[:split], own[split + 1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--pose", choices=["live", "extended", "curl", "contact"], default="live")
    ap.add_argument("--shots", default="")
    ap.add_argument("--frames", default="90,180,270")
    ap.add_argument("--focus", default="20:75:0")
    args = ap.parse_args(own)
    args.root = args.root.resolve()
    sys.path[:0] = [str(args.root), str(args.root / "src"), str(args.root / "cpp" / "build" / "Release")]
    os.chdir(args.root)
    import pyray as rl
    if args.benchmark:
        return benchmark(rl, args)
    import renderer.unit_model_renderer as umr
    original_rl = rl
    state = dict(frame=0, calls=0, poses=set(), drivers={})

    class RendererRaylib:
        """Proxy only in the imported renderer module, never installed pyray."""
        def __getattr__(self, name):
            return getattr(original_rl, name)

        def update_model_animation(self, model, unused_clip, unused_frame):
            key = int(rl.ffi.cast("uintptr_t", model.bones))
            if key not in state["drivers"]:
                state["drivers"][key] = LiveArm(original_rl, model)
            seconds = state["frame"] / 30 if args.shots else rl.get_time()
            controls = arm.cycle_weights(seconds) if args.pose == "live" else {"extended": (0, 0), "curl": (1, 0), "contact": (0, 1)}[args.pose]
            state["drivers"][key].update(*controls)
            state["calls"] += 1
            state["poses"].add(tuple(round(x, 5) for x in controls))

    umr.rl = RendererRaylib()
    original_draw = umr.UnitModelRenderer.draw_units

    def draw_counted(self, units, *a, **k):
        # Game uses one shared render clock for players/zombies. Count once.
        clock = k.get("clock", a[1] if len(a) > 1 else None)
        if clock is None or clock != state.get("clock"):
            state["frame"] += 1
            state["clock"] = clock
        return original_draw(self, units, *a, **k)

    umr.UnitModelRenderer.draw_units = draw_counted
    helper = args.root / "art" / "characters" / "charkit" / "preview_in_game.py"
    helper_args = ["--root", str(args.root), "--model", str(MODEL)]
    if args.shots:
        helper_args += ["--shots", args.shots, "--frames", args.frames, "--focus", args.focus,
                        "--mouse", "640,350"]
    sys.argv = [str(helper), *helper_args, "--", *(passthrough or ["--level", "playground"])]
    exit_code = 0
    try:
        runpy.run_path(str(helper), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        report = dict(root=str(args.root), root_revision=subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
                      procedural_skin_calls=state["calls"], distinct_control_pairs=len(state["poses"]),
                      pose_mode=args.pose, game_exit_code=exit_code,
                      note="Real main.py via root charkit preview + tools/e2e_drive.py. Only prototype pose arguments changed in process; existing UnitModelRenderer, lighting, shader, camera and shadow draw are used.")
        (HOME / "prototype" / "game_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
        print("ARM_GAME_EVIDENCE", json.dumps(report))
        if args.shots and exit_code == 0 and (not state["calls"] or (args.pose == "live" and len(state["poses"]) < 2)):
            raise SystemExit("Game ran without the requested live arm deformation")


if __name__ == "__main__":
    main()
