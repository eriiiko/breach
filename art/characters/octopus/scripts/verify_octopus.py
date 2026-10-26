"""Property checks for the whole-octopus prototype (controls, rig, both LODs).

Standalone prototype checks, not the game test suite; run with Breach's Python:
    python verify_octopus.py
Every check names the property it protects and what must break it. No exact
vertex/triangle counts or poses are pinned: counts are reported, not asserted.
"""
from collections import Counter
from dataclasses import fields
import json
import math
from pathlib import Path
import time

import numpy as np

import arm_math as arm
import octopus_math as om

RNG = np.random.default_rng(20261027)
FAILS = []
OUT = Path(__file__).resolve().parent.parent / "prototype" / "octopus"


def random_arm(rng, edge=False):
    vals = {}
    for f in fields(om.ArmParams):
        lo, hi = om.ArmParams.RANGES[f.name]
        vals[f.name] = float(rng.choice([lo, hi])) if edge and rng.random() < .5 else float(rng.uniform(lo, hi))
    return om.ArmParams(**vals)


def random_body(rng, edge=False):
    vals = {}
    for f in fields(om.BodyParams):
        lo, hi = om.BodyParams.RANGES[f.name]
        vals[f.name] = float(rng.choice([lo, hi])) if edge and rng.random() < .5 else float(rng.uniform(lo, hi))
    return om.BodyParams(**vals)


def arm_slice(k, lod):
    o = om.arm_bone_offset(k, lod)
    return slice(o, o + om.arm_joint_count(lod))


def check_controls(report):
    """Out-of-range or non-finite controls are refused (a controller bug must
    fail loudly, never extrapolate); determinism; frames are rotations; fixed
    segment length; game joints ARE hero joints (same curve, subsampled)."""
    for bad in (om.ArmParams(curl=1.2), om.ArmParams(bend_dir=-1.5), om.ArmParams(lift=float("nan"))):
        try:
            om.pose_octopus(om.BodyParams(), [bad] + [om.ArmParams()] * 7)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range control accepted")
    worst_len, worst_orth = 0., 0.
    for trial in range(300):
        body, arms = random_body(RNG, trial % 2 == 0), [random_arm(RNG, trial % 2 == 0) for _ in range(8)]
        p, r = om.pose_octopus(body, arms, "hero")
        p2, r2 = om.pose_octopus(body, arms, "hero")
        assert np.array_equal(p, p2) and np.array_equal(r, r2), "non-deterministic pose"
        assert np.isfinite(p).all() and np.isfinite(r).all()
        orth = np.abs(np.einsum("nij,nkj->nik", r, r) - np.eye(3)).max()
        assert orth < 1e-10 and np.abs(np.linalg.det(r) - 1).max() < 1e-10, "not a rotation"
        worst_orth = max(worst_orth, float(orth))
        pg, rg = om.pose_octopus(body, arms, "game")
        for k in range(8):
            seg = np.linalg.norm(np.diff(p[arm_slice(k, "hero")], axis=0), axis=1)
            err = float(np.abs(seg - arm.STEP * om.OCTOPUS_SCALE).max())
            worst_len = max(worst_len, err)
            assert err < 1e-12, "arm segment length changed"
            assert np.array_equal(pg[arm_slice(k, "game")], p[arm_slice(k, "hero")][::om.GAME_EVERY])
            assert np.array_equal(rg[arm_slice(k, "game")], r[arm_slice(k, "hero")][::om.GAME_EVERY])
    report["controls"] = dict(random_poses=300, segment_length_max_error_m=worst_len,
                              rotation_orthonormality_max_error=worst_orth)


def check_independence(report):
    """Arm i's bones depend only on arm i's params + the shared body frame
    (body_pitch, crown_spread). Coupling arms (a shared curve, a leaked loop
    variable) must break this; so must a body control other than the two
    documented shared ones reaching the arms."""
    for trial in range(60):
        body = random_body(RNG)
        arms = [random_arm(RNG) for _ in range(8)]
        p0, r0 = om.pose_octopus(body, arms)
        j = int(RNG.integers(8))
        moved = list(arms)
        moved[j] = random_arm(RNG, edge=True)
        p1, r1 = om.pose_octopus(body, moved)
        for i in range(8):
            same = np.array_equal(p0[arm_slice(i, "hero")], p1[arm_slice(i, "hero")]) and \
                np.array_equal(r0[arm_slice(i, "hero")], r1[arm_slice(i, "hero")])
            assert same == (i != j) or (i == j and arms[j] == moved[j]), f"arm {i} moved when arm {j} changed"
        b2 = om.BodyParams(**{**vars(body), "mantle_squeeze": float(RNG.uniform(0, 1)),
                              "mantle_pitch": float(RNG.uniform(-1, 1)), "siphon_yaw": float(RNG.uniform(-1, 1)),
                              "siphon_pitch": float(RNG.uniform(-1, 1))})
        p2, r2 = om.pose_octopus(b2, arms)
        n = len(om.BODY_BONES)
        assert np.array_equal(p0[n:], p2[n:]) and np.array_equal(r0[n:], r2[n:]), "non-shared body control moved an arm"
    report["independence"] = "60 random trials: changing arm j never moves arm i != j; mantle/siphon controls never move arms"


def check_continuity(report):
    """Every control is continuous over its whole range (no jump, no flip at the
    edges): sweeping one control in 400 steps, the largest per-step joint
    displacement stays within 4x the median step, and bend_dir's two ends are
    the same pose (it is periodic). A branch switch, a Frenet flip or a wrap
    discontinuity must break it."""
    out = {}
    for group, cls in (("arm", om.ArmParams), ("body", om.BodyParams)):
        for f in fields(cls):
            lo, hi = cls.RANGES[f.name]
            worst_ratio, worst_rot = 0., 0.
            for trial in range(6):
                body, arms = random_body(RNG), [random_arm(RNG) for _ in range(8)]
                prev, steps, rsteps = None, [], []
                for v in np.linspace(lo, hi, 401):
                    if group == "arm":
                        a = list(arms)
                        a[3] = om.ArmParams(**{**vars(arms[3]), f.name: float(v)})
                        p, r = om.pose_octopus(body, a)
                    else:
                        p, r = om.pose_octopus(om.BodyParams(**{**vars(body), f.name: float(v)}), arms)
                    if prev is not None:
                        steps.append(np.linalg.norm(p - prev[0], axis=1).max())
                        cosang = (np.einsum("nij,nij->n", r, prev[1]) - 1) / 2
                        rsteps.append(float(np.arccos(np.clip(cosang, -1, 1)).max()))
                    prev = (p, r)
                steps = np.array(steps)
                ratio = float(steps.max() / max(np.median(steps), 1e-12))
                worst_ratio, worst_rot = max(worst_ratio, ratio), max(worst_rot, max(rsteps))
                assert ratio < 4.0, f"{group}.{f.name}: jump (max/median step {ratio:.2f})"
                assert max(rsteps) < 0.06, f"{group}.{f.name}: frame flip ({max(rsteps):.3f} rad per step)"
            out[f"{group}.{f.name}"] = dict(max_over_median_step=round(worst_ratio, 3),
                                            max_frame_step_rad=round(worst_rot, 5))
    a = [om.ArmParams(curl=.7, bend_dir=1.0, twist=.4, tip_curl=-.3)] * 8
    b = [om.ArmParams(curl=.7, bend_dir=-1.0, twist=.4, tip_curl=-.3)] * 8
    pa, ra = om.pose_octopus(om.BodyParams(), a)
    pb, rb = om.pose_octopus(om.BodyParams(), b)
    assert np.abs(pa - pb).max() < 1e-12 and np.abs(ra - rb).max() < 1e-12, "bend_dir not periodic"
    report["continuity"] = out


def mesh_checks(d, name, report):
    """Closed manifold surface (incl. the junction: arm rings are SHARED with the
    crown, so a cut cylinder or a gap breaks the edge count); normalized,
    nonnegative <= 4-influence weights; bind skin reproduces the source mesh."""
    edges = Counter()
    for f in d["faces"]:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert set(edges.values()) == {2}, f"{name}: open or non-manifold edges {Counter(edges.values())}"
    assert np.isfinite(d["vertices"]).all()
    w = d["weights"]
    assert w.min() >= 0 and np.abs(w.sum(axis=1) - 1).max() < 1e-9
    p, r = om.pose_octopus(om.BodyParams(), None, d["lod"])
    bind = om.skin(d["vertices"], d["joints"], w, d["bind"], p, r)
    assert np.abs(bind - d["vertices"] * om.OCTOPUS_SCALE).max() < 1e-12, "bind pose does not reproduce the mesh"
    # junction ring vertices are referenced by crown faces (shared, not duplicated)
    lo, hi = d["crown_range"]
    crown_refs = {i for f in d["faces"] for i in f if any(lo <= j < hi for j in f)}
    for k, (a, b) in enumerate(d["arm_ranges"]):
        ring = set(range(a, a + d["sectors"]))
        assert ring <= crown_refs, f"{name}: arm {k} junction ring not stitched into the crown"
    tris = sum(len(f) - 2 for f in d["faces"])
    report[f"{name}_mesh"] = dict(vertices=len(d["vertices"]), triangles=tris, joints=len(d["bones"]),
                                  max_influences=int((w > 0).sum(axis=1).max()))
    return np.array([[f[0], f[i], f[i + 1]] for f in d["faces"] for i in range(1, len(f) - 1)])


def tri_area(v, t):
    return np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1) / 2


def check_deformation(game, hero, report):
    """Worst deformation over random poses spanning the full control ranges,
    measured on deformed surfaces. Reports the minimum triangle-area ratio of
    the game skin split by region (arm tubes / crown+body), the arm tube cross
    sections on the hero (the proof's measure), and the game-LOD rig error
    against the hero rig on the same surface points. Gross collapse or
    inflation must fail; the numbers themselves are recorded, not pinned."""
    tg = check_deformation.tg
    rest_area = tri_area(game["vertices"], tg)
    region = np.zeros(len(tg), int)
    lo, hi = game["crown_range"]
    region[np.isin(tg, np.arange(lo, hi)).any(axis=1)] = 1      # crown, body, web and the seam faces
    rest_n = np.cross(game["vertices"][tg[:, 1]] - game["vertices"][tg[:, 0]],
                      game["vertices"][tg[:, 2]] - game["vertices"][tg[:, 0]])
    dom = game["joints"][tg[:, 0], np.argmax(game["weights"][tg[:, 0]], axis=1)]

    def flips(v, r):
        n = np.cross(v[tg[:, 1]] - v[tg[:, 0]], v[tg[:, 2]] - v[tg[:, 0]])
        ref = np.einsum("nij,nj->ni", r[dom], rest_n)
        return int(((n * ref).sum(axis=1) < 0).sum())
    sectors = hero["sectors"]
    rings = (om.HERO_RINGS - 1 - om.JUNCTION_RING)
    hv = hero["vertices"]

    def section_areas(v, k):
        a = hero["arm_ranges"][k][0]
        rr = v[a:a + rings * sectors].reshape(rings, sectors, 3)
        return np.linalg.norm(np.cross(rr, np.roll(rr, -1, axis=1)).sum(axis=1), axis=1) / 2

    rest_sec = [section_areas(hv, k) for k in range(8)]
    # LOD error: skin the GAME's bind vertices with the hero rig (hero weights).
    gv = game["vertices"]
    hero_w = {}
    for k, (a, b) in enumerate(game["arm_ranges"]):
        x = ((gv[a:b] - om.MOUNT_POINTS[k]) @ om.MOUNT_FRAMES[k])[:, 0]
        f = np.clip(x / arm.STEP, 0, arm.SEGMENTS)
        j0 = np.minimum(f.astype(int), arm.SEGMENTS - 1)
        t = f - j0
        o = om.arm_bone_offset(k, "hero")
        jj = np.zeros((b - a, 4), int); ww = np.zeros((b - a, 4))
        jj[:, 0], jj[:, 1], ww[:, 0], ww[:, 1] = o + j0, o + j0 + 1, 1 - t, t
        hero_w[k] = (a, b, jj, ww)
    worst = dict(arm=1., crown=1., crown_max=1., section_min=1., section_max=1., lod_err=0., flips=0, flip_poses=0)
    samples = 120
    t0 = time.time()
    for trial in range(samples):
        edge = trial % 3 == 0
        body, arms = random_body(RNG, edge), [random_arm(RNG, edge) for _ in range(8)]
        p, r = om.pose_octopus(body, arms, "game")
        v = om.skin(gv, game["joints"], game["weights"], game["bind"], p, r)
        ratio = tri_area(v, tg) / rest_area
        worst["arm"] = min(worst["arm"], float(ratio[region == 0].min()))
        worst["crown"] = min(worst["crown"], float(ratio[region == 1].min()))
        worst["crown_max"] = max(worst["crown_max"], float(ratio[region == 1].max()))
        nf = flips(v, r)
        worst["flips"] = max(worst["flips"], nf)
        worst["flip_poses"] += nf > 0
        ph, rh = om.pose_octopus(body, arms, "hero")
        if trial % 4 == 0:
            vh = om.skin(hv, hero["joints"], hero["weights"], hero["bind"], ph, rh)
            for k in range(8):
                s = section_areas(vh, k) / rest_sec[k]
                worst["section_min"] = min(worst["section_min"], float(s.min()))
                worst["section_max"] = max(worst["section_max"], float(s.max()))
        for k, (a, b, jj, ww) in hero_w.items():
            vh = om.skin(gv[a:b], jj, ww, hero["bind"], ph, rh)
            worst["lod_err"] = max(worst["lod_err"], float(np.linalg.norm(vh - v[a:b], axis=1).max()))
    # Thresholds: gross collapse only. The crown/web minimum is a fold-prone
    # sliver at the arm-root top under full lift (measured 0.04); a regression
    # to a rigid root pivot measured 0.007 and must fail.
    assert worst["arm"] > 0.5, f"arm tube triangles collapse ({worst['arm']:.3f})"
    # Full-range random combinations (root pitch up to 65 deg on all arms plus
    # swing/curl) DO pinch slivers in the crown: recorded, a known limit. The
    # regression guard is the single-control extreme: full lift alone.
    for lift in (-1.0, 1.0):
        p, r = om.pose_octopus(om.BodyParams(), [om.ArmParams(lift=lift)] * 8, "game")
        ratio = tri_area(om.skin(gv, game["joints"], game["weights"], game["bind"], p, r), tg) / rest_area
        assert ratio[region == 1].min() > 0.02, "full lift folds the crown (rigid root pivot regression)"
    assert worst["section_min"] > 0.85 and worst["section_max"] < 1.02, "arm cross-section collapse/inflation"
    assert worst["lod_err"] < 0.03 * om.OCTOPUS_SCALE, f"game rig far from hero ({worst['lod_err']:.4f} m)"
    report["deformation_random_full_range"] = dict(
        poses=samples, game_arm_triangle_area_ratio_min=worst["arm"],
        game_crown_triangle_area_ratio_min=worst["crown"], game_crown_triangle_area_ratio_max=worst["crown_max"],
        hero_arm_section_area_ratio_min=worst["section_min"], hero_arm_section_area_ratio_max=worst["section_max"],
        game_vs_hero_rig_max_deviation_m=worst["lod_err"],
        inverted_triangles_worst_pose=worst["flips"], poses_with_inverted_triangles=int(worst["flip_poses"]),
        seconds=round(time.time() - t0, 1))
    # Named demonstration poses and the swim loop: the shown poses.
    shown = dict(arm=1., crown=1., flips=0)
    poses = list(om.named_poses().values()) + [om.swim_cycle(t) for t in np.linspace(0, om.SWIM_PERIOD, 25)]
    for body, arms in poses:
        p, r = om.pose_octopus(body, arms, "game")
        ratio = tri_area(om.skin(gv, game["joints"], game["weights"], game["bind"], p, r), tg) / rest_area
        shown["arm"] = min(shown["arm"], float(ratio[region == 0].min()))
        shown["crown"] = min(shown["crown"], float(ratio[region == 1].min()))
        shown["flips"] = max(shown["flips"], flips(om.skin(gv, game["joints"], game["weights"], game["bind"], p, r), r))
    report["deformation_shown_poses"] = dict(poses=len(poses), game_arm_triangle_area_ratio_min=shown["arm"],
                                             game_crown_triangle_area_ratio_min=shown["crown"],
                                             inverted_triangles_max=shown["flips"])
    # Measured 2026-10-27: 4-8 inverted slivers (of 12k) in the underside/web
    # between the back arms when the crown closes (swim) or the walk swings the
    # back arms -- a known, reported limit; a gross fold must still fail.
    if shown["flips"] > 0.001 * len(tg):
        FAILS.append("a shown pose folds the skin")
    if not (shown["arm"] > 0.5 and shown["crown"] > 0.02):
        FAILS.append("a shown pose collapses triangles")
    # swim loop continuity across its boundary
    a = om.pose_octopus(*om.swim_cycle(om.SWIM_PERIOD - 1e-7))[0]
    b = om.pose_octopus(*om.swim_cycle(1e-7))[0]
    assert np.abs(a - b).max() < 1e-5, "swim loop not continuous"


def main():
    report = {}
    t0 = time.time()
    check_controls(report)
    check_independence(report)
    check_continuity(report)
    game, hero = om.assemble("game"), om.assemble("hero")
    check_deformation.tg = mesh_checks(game, "game", report)
    mesh_checks(hero, "hero", report)
    check_deformation(game, hero, report)
    report["seconds"] = round(time.time() - t0, 1)
    report["limits"] = ("Random poses span every control's full range including edges; no self-collision, "
                        "arm-arm or arm-body intersection, ground contact or volume preservation is checked "
                        "or solved. Crown/body areas are measured on the GAME skin; the hero shares its weights' "
                        "construction. Thresholds are prototype acceptance, not a production spec.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "property_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert not FAILS, FAILS
    print("OCTOPUS_PROPERTIES_OK")


if __name__ == "__main__":
    main()
