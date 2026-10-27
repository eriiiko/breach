"""Whole-octopus client of Breach's existing skin/render seam (builds on runtime_arm.py).

Same method as the one-arm proof: --root is a read-only Breach checkout; the
renderer module gets a process-local raylib proxy whose update_model_animation
receives a freshly evaluated one-frame pose from octopus_math.pose_octopus,
then calls the real native raylib skinning. No production file is touched.

    python runtime_octopus.py --root <breach> -- --level playground          (play)
    python runtime_octopus.py --root <breach> --shots <prefix> --frames 90,200 -- --level playground
    python runtime_octopus.py --root <breach> --benchmark --samples 120
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
import octopus_math as om

HOME = Path(__file__).resolve().parent.parent
MODEL = HOME / "prototype" / "octopus" / "octopus_game.glb"
REPORT = HOME / "prototype" / "octopus"


class LiveOctopus:
    """One-frame ModelAnimation over the loaded octopus skeleton (bones by name).

    The quaternion composition (pose delta x bind rotation) is runtime_arm.LiveArm's.

    Bone SCALE (the arm stretch's thinning, #106). Measured on pyray 5.5.0.4:
    raylib's CPU skin applies a frame pose's scale AFTER the bone's rotation and
    translation, in model axes, i.e. v -> s * (R v + t) about the model origin,
    not about the bone. For the UNIFORM scale the octopus uses that is still
    exact if the translation is pre-compensated: writing translation
    p/s + (s - 1) R_def b (b = the bone's bind head) gives s R_def (v - b) + p,
    the reference skin. The benchmark's parity gate (stretched poses included)
    fails if a raylib update changes this convention.
    """

    def __init__(self, rl, model):
        self.rl, self.model = rl, model
        self.pose_array = rl.ffi.new("Transform[]", model.boneCount)
        self.frames = rl.ffi.new("Transform*[]", [self.pose_array])
        self.animation = rl.ffi.new("ModelAnimation *")
        self.animation.boneCount = model.boneCount
        self.animation.bones = model.bones
        self.animation.framePoses = self.frames
        for f in ("frameCount", "keyframeCount"):
            if hasattr(self.animation, f):
                setattr(self.animation, f, 1)
                break
        order = {n: i for i, n in enumerate(om.bones("game"))}
        self.map = []
        self.bind_head = np.zeros((len(order), 3))
        for i in range(model.boneCount):
            name = rl.ffi.string(model.bones[i].name).decode()
            self.pose_array[i] = model.bindPose[i]
            q = model.bindPose[i].rotation
            t = model.bindPose[i].translation
            self.map.append((i, order[name], (q.x, q.y, q.z, q.w)))
            self.bind_head[order[name]] = (t.x, t.y, t.z)
        if sorted(j for _, j, _ in self.map) != list(range(len(order))):
            raise ValueError("loaded skeleton does not match octopus_math.bones('game')")
        self.scaled = np.zeros(len(order), bool)       # bones whose written scale is not 1

    def write_pose(self, points, rotations, scales=None):
        pa = self.pose_array
        if scales is not None and (scales != 1.0).any():
            # raylib's model-axis scale about the origin, pre-compensated (class doc)
            rb = np.einsum("nij,nj->ni", rotations, self.bind_head)
            points = points / scales[:, None] + (scales - 1)[:, None] * rb
        else:
            scales = None
        for bone, j, (bx, by, bz, bw) in self.map:
            tr = pa[bone]
            p = points[j]
            tr.translation.x, tr.translation.y, tr.translation.z = float(p[0]), float(p[1]), float(p[2])
            x, y, z, w = arm.quaternion(rotations[j])
            tr.rotation.x = w * bx + x * bw + y * bz - z * by
            tr.rotation.y = w * by - x * bz + y * bw + z * bx
            tr.rotation.z = w * bz + x * by - y * bx + z * bw
            tr.rotation.w = w * bw - x * bx - y * by - z * bz
            s = 1.0 if scales is None else float(scales[j])
            if s != 1.0 or self.scaled[j]:
                tr.scale.x = tr.scale.y = tr.scale.z = s
                self.scaled[j] = s != 1.0

    def update(self, body, arms):
        self.write_pose(*om.pose_octopus(body, arms, "game", with_scale=True))
        self.rl.update_model_animation(self.model, self.animation[0], 0)


def stretch_demo(seconds):
    """demo_cycle with the two front arms reaching out and back (stretch 0..1..0 over 3 s)."""
    body, arms = om.demo_cycle(seconds)
    st = 0.5 - 0.5 * np.cos(2 * np.pi * seconds / 3.0)
    for k in (0, 4):
        arms[k] = om.ArmParams(**{**vars(arms[k]), "stretch": float(st)})
    return body, arms


def revision(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def benchmark(rl, args):
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(128, 128, "Octopus skin benchmark (hidden)")
    model = rl.load_model(str(MODEL))
    drv = LiveOctopus(rl, model)
    geo = np.load(REPORT / "octopus_game_geometry.npz")
    if "scale" in geo.files and abs(float(geo["scale"]) - om.OCTOPUS_SCALE) > 1e-12:
        raise SystemExit(f"the GLB was built at scale {float(geo['scale'])}; active size {om.SIZE} is {om.OCTOPUS_SCALE}")
    src = geo["vertices"] * om.OCTOPUS_SCALE
    lookup = {tuple(np.round(v, 5)): i for i, v in enumerate(src)}
    maps, bind_err = [], 0.
    for m in range(model.meshCount):
        mesh = model.meshes[m]
        vv = np.frombuffer(rl.ffi.buffer(mesh.vertices, mesh.vertexCount * 12), dtype=np.float32).reshape(-1, 3)
        idx = np.array([lookup.get(tuple(np.round(v, 5)), -1) for v in vv])
        for i in np.nonzero(idx < 0)[0]:
            idx[i] = int(np.argmin(((src - vv[i]) ** 2).sum(1)))
        bind_err = max(bind_err, float(np.linalg.norm(src[idx] - vv, axis=1).max()))
        maps.append(idx)
    errors = []
    rng = np.random.default_rng(106)
    poses = [om.demo_cycle(t) for t in np.linspace(0, om.WALK_PERIOD + 2 * om.SWIM_PERIOD, 31)] + \
        list(om.named_poses().values()) + [stretch_demo(t) for t in np.linspace(0, 3, 12)] + \
        [(om.BodyParams(), [om.ArmParams(curl=float(rng.uniform(0, 1)), bend_dir=float(rng.uniform(-1, 1)),
                                         lift=float(rng.uniform(-1, 1)), stretch=float(rng.uniform(0, 1)))
                            for _ in range(8)]) for _ in range(6)]
    for body, arms in poses:
        drv.update(body, arms)
        p, r, sc = om.pose_octopus(body, arms, "game", with_scale=True)
        exp = om.skin(geo["vertices"], geo["joints"], geo["weights"], geo["bind"], p, r, sc)
        for m, idx in enumerate(maps):
            mesh = model.meshes[m]
            act = np.frombuffer(rl.ffi.buffer(mesh.animVertices, mesh.vertexCount * 12), dtype=np.float32).reshape(-1, 3)
            errors.append(float(np.linalg.norm(act - exp[idx], axis=1).max()))
    assert bind_err < 2e-6, bind_err
    assert max(errors) < 1e-5, max(errors)
    for _ in range(20):
        drv.update(*om.demo_cycle(.3))
    batches = {}
    for source, count in [(src_, n_) for src_ in ("demo", "demo_two_arms_stretching") for n_ in (1, 4)]:
        cycle = om.demo_cycle if source == "demo" else stretch_demo
        samples = []
        for s in range(args.samples):
            ev = wr = sk = 0.
            start = time.perf_counter_ns()
            for i in range(count):
                t0 = time.perf_counter_ns()
                body, arms = cycle(s * .017 + i * .37)
                p, r, sc = om.pose_octopus(body, arms, "game", with_scale=True)
                t1 = time.perf_counter_ns()
                drv.write_pose(p, r, sc)
                t2 = time.perf_counter_ns()
                rl.update_model_animation(model, drv.animation[0], 0)
                t3 = time.perf_counter_ns()
                ev += (t1 - t0) / 1e6
                wr += (t2 - t1) / 1e6
                sk += (t3 - t2) / 1e6
            samples.append([ev, wr, sk, (time.perf_counter_ns() - start) / 1e6])
        v = np.array(samples)
        batches[f"{source}/{count}"] = {n: dict(median_ms=float(np.median(v[:, i])), p95_ms=float(np.percentile(v[:, i], 95)))
                               for i, n in enumerate(("controls_and_pose_octopus", "write_joint_transforms",
                                                      "raylib_skin_and_upload", "total"))}
    rep = dict(root=str(args.root), root_revision=revision(args.root), python=sys.version, platform=platform.platform(),
               processor=platform.processor(), model=MODEL.name, model_meshes=model.meshCount,
               model_joints=model.boneCount,
               loaded_vertices=sum(model.meshes[i].vertexCount for i in range(model.meshCount)),
               loaded_triangles=sum(model.meshes[i].triangleCount for i in range(model.meshCount)),
               bind_vertex_max_error_m=bind_err, raylib_vs_numpy_max_error_m=max(errors), parity_poses=len(poses),
               samples_per_batch=args.samples, timings_octopi=batches, size=om.size_report(),
               methodology=("perf_counter_ns wall time, 20 warmups; sequential independently phased whole-animal "
                            "poses on one shared loaded model (the renderer's shared-model seam). Skin call includes "
                            "driver buffer submission, not GPU completion. Excludes drawing, shading, shadows, "
                            "simulation. CPU timings are not whole-game FPS."))
    (REPORT / "runtime_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))
    rl.unload_model(model)
    rl.close_window()


def main():
    own, passthrough = sys.argv[1:], []
    if "--" in own:
        k = own.index("--")
        own, passthrough = own[:k], own[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--pose", default="live", choices=["live", *om.named_poses().keys()])
    ap.add_argument("--shots", default="")
    ap.add_argument("--frames", default="90,200")
    ap.add_argument("--focus", default="20:30:0")
    ap.add_argument("--model", type=Path, default=MODEL, help="a game GLB, e.g. a skin variant (#113)")
    ap.add_argument("--evidence", type=Path, default=REPORT / "game_evidence.json")
    args = ap.parse_args(own)
    args.model, args.evidence = args.model.resolve(), args.evidence.resolve()
    args.root = args.root.resolve()
    sys.path[:0] = [str(args.root), str(args.root / "src"), str(args.root / "cpp" / "build" / "Release")]
    os.chdir(args.root)
    import pyray as rl
    if args.benchmark:
        return benchmark(rl, args)
    import renderer.unit_model_renderer as umr
    real = rl
    state = dict(frame=0, calls=0, poses=set(), drivers={})

    class RendererRaylib:
        """Proxy only inside the imported renderer module, never installed pyray."""
        def __getattr__(self, name):
            return getattr(real, name)

        def update_model_animation(self, model, unused_clip, unused_frame):
            key = int(real.ffi.cast("uintptr_t", model.bones))
            if key not in state["drivers"]:
                state["drivers"][key] = LiveOctopus(real, model)
            seconds = state["frame"] / 30 if args.shots else real.get_time()
            controls = om.demo_cycle(seconds) if args.pose == "live" else om.named_poses()[args.pose]
            state["drivers"][key].update(*controls)
            state["calls"] += 1
            state["poses"].add(round(seconds, 3))

    umr.rl = RendererRaylib()
    draw = umr.UnitModelRenderer.draw_units

    def draw_counted(self, units, *a, **k):
        clock = k.get("clock", a[1] if len(a) > 1 else None)
        if clock is None or clock != state.get("clock"):
            state["frame"] += 1
            state["clock"] = clock
        return draw(self, units, *a, **k)

    umr.UnitModelRenderer.draw_units = draw_counted
    helper = args.root / "art" / "characters" / "charkit" / "preview_in_game.py"
    hargs = ["--root", str(args.root), "--model", str(args.model)]
    if args.shots:
        hargs += ["--shots", args.shots, "--frames", args.frames, "--focus", args.focus, "--mouse", "640,350"]
    sys.argv = [str(helper), *hargs, "--", *(passthrough or ["--level", "playground"])]
    code = 0
    try:
        runpy.run_path(str(helper), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        raise
    except BaseException:
        code = 1
        raise
    finally:
        rep = dict(root=str(args.root), root_revision=revision(args.root), procedural_skin_calls=state["calls"],
                   distinct_pose_times=len(state["poses"]), pose_mode=args.pose, game_exit_code=code,
                   note="Real main.py via root charkit preview + tools/e2e_drive.py; only this prototype's pose "
                        "argument changes in process. UnitModelRenderer, lighting, shader, camera, shadows unchanged.")
        rep["model"] = args.model.name
        args.evidence.write_text(json.dumps(rep, indent=2) + "\n")
        print("OCTOPUS_GAME_EVIDENCE", json.dumps(rep))
        if args.shots and code == 0 and state["calls"] < 2:
            raise SystemExit("Game ran without live octopus posing")


if __name__ == "__main__":
    main()
