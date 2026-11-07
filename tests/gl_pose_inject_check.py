"""GL checks of the pose layer's CPU-inject skinner (#162; design
docs/animation_milestones_design_2026-11-02.md §4.5, property 12).

Runs in ITS OWN PROCESS with a hidden raylib window (a GL context), never
inside the pytest process: ``tests/test_pose_inject_gl.py`` spawns it and
asserts on its last line (the ``gl_*_check.py`` convention). Prints one
``[check] NAME: PASS|FAIL detail`` line per check and finally
``POSE_INJECT_GL_RESULT: PASS`` or ``... FAIL``.

Each check names the PROPERTY it protects and the change that must break it:

  inject_is_raylibs_own_frame -- on THIS machine's binding, writing raylib's
      own frame-k global pose into the skinner's scratch clip and skinning
      reproduces ``update_model_animation(model, clip, k)``'s animVertices
      and animNormals BIT FOR BIT, for every shipped look, several clips and
      frames. Breaks if the scratch clip's field names (5.x framePoses / 6.x
      keyframePoses), its bone table, its frame count or the Transform row
      layout is wrong for the binding.
  composed_pose_skins_like_raylib -- the pose layer's own composed pose of a
      clip raylib samples on its keys (a frame whose bones all hold their
      floor key on the 5.5 loader; frame 0 on any binding) skins to within
      a millimetre-scale bound of raylib's own vertices for that frame.
      Breaks if the composed global pose and the mesh disagree (bone order,
      model vs local space, a wrong root).
  clips_untouched_by_injection -- after injecting, raylib's own clips still
      skin exactly as before. Breaks if the scratch clip aliases and
      overwrites a loaded clip's pose buffers.
  refuses_a_foreign_bone_table -- preparing a look against a bone list that
      is not raylib's raises. Breaks if a mismatched skeleton is accepted
      (its rows would land on the wrong bones).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pyray as rl  # noqa: E402

RESULTS = []


def check(fn) -> None:
    name = fn.__name__
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        RESULTS.append((name, True, f"{detail} [{time.perf_counter() - t0:.1f}s]"))
    except Exception as e:  # noqa: BLE001 - every failure is a FAIL line
        RESULTS.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))



def _looks():
    from config import CFG
    from renderer.unit_looks import all_look_names, look_path, looks_table
    out = []
    for name in all_look_names(looks_table(CFG)):
        path = str(look_path(name).resolve())
        model = rl.load_model(path)
        n = rl.ffi.new("int *", 0)
        anims = rl.load_model_animations(path, n)
        names = {rl.ffi.string(anims[i].name).decode(): i for i in range(n[0])}
        out.append((name, path, model, anims, n[0], names))
    return out


def _verts(model):
    ffi = rl.ffi
    v, nrm = [], []
    for m in range(model.meshCount):
        mesh = model.meshes[m]
        nv = mesh.vertexCount
        v.append(np.frombuffer(ffi.buffer(mesh.animVertices, nv * 12), np.float32).copy())
        nrm.append(np.frombuffer(ffi.buffer(mesh.animNormals, nv * 12), np.float32).copy())
    return np.concatenate(v), np.concatenate(nrm)


def _frame_pose(anim, k, poses_field):
    J = int(anim.boneCount)
    return np.frombuffer(rl.ffi.buffer(getattr(anim, poses_field)[k], J * 40),
                         np.float32).reshape(J, 10).copy()


LOOKS = []


def inject_is_raylibs_own_frame():
    from renderer.pose_skinner import CpuInjectSkinner, anim_pose_fields
    from renderer.unit_model_renderer import anim_frame_count
    count_f, poses_f = anim_pose_fields(rl.ffi)
    sk = CpuInjectSkinner(rl)
    compared = 0
    for name, _, model, anims, n, names in LOOKS:
        sk.prepare(name, model, anims[0])
        for clip in list(names)[:: max(1, n // 6)]:
            a = anims[names[clip]]
            fc = anim_frame_count(a)
            for k in sorted({0, fc // 3, fc - 1}):
                rl.update_model_animation(model, a, k)
                ref_v, ref_n = _verts(model)
                rl.update_model_animation(model, anims[(names[clip] + 1) % n], 0)  # scramble
                sk.apply(name, model, _frame_pose(a, k, poses_f))
                got_v, got_n = _verts(model)
                assert np.array_equal(ref_v, got_v), (name, clip, k, float(np.abs(ref_v - got_v).max()))
                assert np.array_equal(ref_n, got_n), (name, clip, k)
                compared += 1
    return f"{compared} frames bit-identical ({count_f}/{poses_f})"


def composed_pose_skins_like_raylib():
    from config import CFG
    from renderer.pose_layer import PoseLayer, pose_settings
    from renderer.pose_skinner import CpuInjectSkinner, anim_pose_fields
    _, poses_f = anim_pose_fields(rl.ffi)
    settings = pose_settings(CFG)
    walk = settings.states["walk"][settings.layers[0]]
    pl = PoseLayer(settings)
    sk = CpuInjectSkinner(rl)
    worst = 0.0
    for name, path, model, anims, n, names in LOOKS:
        rig = pl.add_look(name, path)
        sk.prepare(name, model, anims[0], rig.bank.skeleton.names)
        bank = rig.bank
        c = bank.index(walk)
        a = anims[names[walk]]
        frames = [0] + ([12, 37] if poses_f == "framePoses" else [])
        for f in frames:
            key = min(int(np.floor(f * 0.017 / bank.key_dt[c] + 1e-4)), bank.n_keys[c] - 1)
            ps = pl.new_state(name, settings.fallback_state)
            ps.clip[:] = c
            ps.time[:] = key * bank.key_dt[c]
            row = pl.compose(name, [ps]).global_trs[0]
            rl.update_model_animation(model, a, f)
            ref_v, _ = _verts(model)
            sk.apply(name, model, row)
            got_v, _ = _verts(model)
            err = float(np.abs(ref_v - got_v).max())
            worst = max(worst, err)
            assert err < 1e-3, (name, f, err)
    return f"max |dv| {worst:.2e} (model units)"


def clips_untouched_by_injection():
    from renderer.pose_skinner import CpuInjectSkinner
    sk = CpuInjectSkinner(rl)
    name, _, model, anims, n, names = LOOKS[0]
    a = anims[0]
    rl.update_model_animation(model, a, 0)
    before, _ = _verts(model)
    sk.prepare(name, model, a)
    J = int(a.boneCount)
    junk = np.zeros((J, 10), np.float32)
    junk[:, 6] = 1.0
    junk[:, 7:] = 1.0
    junk[:, 0] = 5.0
    sk.apply(name, model, junk)
    rl.update_model_animation(model, a, 0)
    after, _ = _verts(model)
    assert np.array_equal(before, after)
    return "loaded clip unchanged"


def refuses_a_foreign_bone_table():
    from renderer.pose_skinner import CpuInjectSkinner
    sk = CpuInjectSkinner(rl)
    name, _, model, anims, n, names = LOOKS[0]
    J = int(anims[0].boneCount)
    try:
        sk.prepare(name, model, anims[0], [f"bone{k}" for k in range(J)])
    except ValueError:
        return "refused"
    raise AssertionError("a foreign bone table was accepted")


def main() -> int:
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(160, 120, "gl_pose_inject_check")
    try:
        LOOKS.extend(_looks())
        for fn in (inject_is_raylibs_own_frame, composed_pose_skins_like_raylib,
                   clips_untouched_by_injection, refuses_a_foreign_bone_table):
            check(fn)
    finally:
        for _, _, model, anims, n, _ in LOOKS:
            rl.unload_model_animations(anims, n)
            rl.unload_model(model)
        rl.close_window()
    ok = all(r[1] for r in RESULTS) and len(RESULTS) == 4
    for name, passed, detail in RESULTS:
        print(f"[check] {name}: {'PASS' if passed else 'FAIL'} {detail}")
    print(f"POSE_INJECT_GL_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
