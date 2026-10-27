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
    fail loudly, never extrapolate); determinism; frames are rotations; every
    segment has exactly the length law's length (rest length from the size row
    x the stretch elongation, #106: a drifting rod, or a stretch leaking into
    the crown segments, breaks it); the joint scales follow the thinning law
    1/sqrt(elongation) and are exactly 1 up to joint 6 (the junction); game
    joints ARE hero joints (same curve, subsampled)."""
    for bad in (om.ArmParams(curl=1.2), om.ArmParams(bend_dir=-1.5), om.ArmParams(lift=float("nan")),
                om.ArmParams(stretch=-0.1), om.ArmParams(stretch=1.01)):
        try:
            om.pose_octopus(om.BodyParams(), [bad] + [om.ArmParams()] * 7)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range control accepted")
    worst_len, worst_orth = 0., 0.
    for trial in range(300):
        body, arms = random_body(RNG, trial % 2 == 0), [random_arm(RNG, trial % 2 == 0) for _ in range(8)]
        p, r, sc = om.pose_octopus(body, arms, "hero", with_scale=True)
        p2, r2 = om.pose_octopus(body, arms, "hero")
        assert np.array_equal(p, p2) and np.array_equal(r, r2), "non-deterministic pose"
        assert np.isfinite(p).all() and np.isfinite(r).all()
        orth = np.abs(np.einsum("nij,nkj->nik", r, r) - np.eye(3)).max()
        assert orth < 1e-10 and np.abs(np.linalg.det(r) - 1).max() < 1e-10, "not a rotation"
        worst_orth = max(worst_orth, float(orth))
        pg, rg = om.pose_octopus(body, arms, "game")
        for k in range(8):
            seg = np.linalg.norm(np.diff(p[arm_slice(k, "hero")], axis=0), axis=1)
            err = float(np.abs(seg - om.arm_segment_lengths(arms[k].stretch) * om.OCTOPUS_SCALE).max())
            worst_len = max(worst_len, err)
            assert err < 1e-12, "arm segment length differs from the length law"
            n0 = om.JUNCTION_JOINT + 1
            assert np.abs(seg[:n0] - arm.STEP * om.OCTOPUS_SCALE).max() < 1e-12, "a crown segment changed length"
            ks = sc[arm_slice(k, "hero")]
            e = 1 + (om.stretch_factor(arms[k].stretch) - 1) * om.RAMP
            assert np.all(ks[:n0 + 1] == 1.0), "a joint at the junction scaled"
            assert np.abs(ks[1:] * np.sqrt(e) - 1).max() < 1e-12, "joint scale off the 1/sqrt thinning law"
            assert np.array_equal(pg[arm_slice(k, "game")], p[arm_slice(k, "hero")][::om.GAME_EVERY])
            assert np.array_equal(rg[arm_slice(k, "game")], r[arm_slice(k, "hero")][::om.GAME_EVERY])
    report["controls"] = dict(random_poses=300, segment_length_max_error_vs_law_m=worst_len,
                              rotation_orthonormality_max_error=worst_orth)


def check_independence(report):
    """Arm i's bones depend only on arm i's params + the shared body frame
    (body_pitch, crown_spread). Coupling arms (a shared curve, a leaked loop
    variable) must break this; so must a body control other than the two
    documented shared ones reaching the arms."""
    for trial in range(60):
        body = random_body(RNG)
        arms = [random_arm(RNG) for _ in range(8)]
        p0, r0, s0 = om.pose_octopus(body, arms, with_scale=True)
        j = int(RNG.integers(8))
        moved = list(arms)
        moved[j] = random_arm(RNG, edge=True)
        p1, r1, s1 = om.pose_octopus(body, moved, with_scale=True)
        for i in range(8):
            same = np.array_equal(p0[arm_slice(i, "hero")], p1[arm_slice(i, "hero")]) and \
                np.array_equal(r0[arm_slice(i, "hero")], r1[arm_slice(i, "hero")]) and \
                np.array_equal(s0[arm_slice(i, "hero")], s1[arm_slice(i, "hero")])
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
    rest_area = tri_area(game["vertices"] * om.OCTOPUS_SCALE, tg)      # posed skins are in world scale
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

    rest_sec = [section_areas(hv * om.OCTOPUS_SCALE, k) for k in range(8)]
    # The stretch thinning law (#106) shrinks a section by the blended joint
    # scale squared: sections are measured AGAINST that expectation, so a
    # collapse beyond the law (or a missing thinning) still fails.
    ring_f = (om.JUNCTION_RING + np.arange(rings)) / (om.HERO_RINGS - 1) * arm.LENGTH / arm.STEP
    ring_a = np.minimum(ring_f.astype(int), arm.SEGMENTS - 1)
    ring_t = ring_f - ring_a

    def expected_sec(sh, k):
        s = sh[arm_slice(k, "hero")]
        return ((1 - ring_t) * s[ring_a] + ring_t * s[ring_a + 1]) ** 2
    # LOD error: skin the GAME's bind vertices with the hero rig (hero weights).
    gv = game["vertices"]
    hero_w = {}
    for k, (a, b) in enumerate(game["arm_ranges"]):
        x = ((gv[a:b] - om.MOUNT_POINTS[k]) @ om.MOUNT_FRAMES[k])[:, 0]
        f = om.arm_joint_coord(om.material_x(x), "hero")
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
        p, r, sc = om.pose_octopus(body, arms, "game", with_scale=True)
        v = om.skin(gv, game["joints"], game["weights"], game["bind"], p, r, sc)
        ratio = tri_area(v, tg) / rest_area
        worst["arm"] = min(worst["arm"], float(ratio[region == 0].min()))
        worst["crown"] = min(worst["crown"], float(ratio[region == 1].min()))
        worst["crown_max"] = max(worst["crown_max"], float(ratio[region == 1].max()))
        nf = flips(v, r)
        worst["flips"] = max(worst["flips"], nf)
        worst["flip_poses"] += nf > 0
        ph, rh, sh = om.pose_octopus(body, arms, "hero", with_scale=True)
        if trial % 4 == 0:
            vh = om.skin(hv, hero["joints"], hero["weights"], hero["bind"], ph, rh, sh)
            for k in range(8):
                s = section_areas(vh, k) / rest_sec[k] / expected_sec(sh, k)
                worst["section_min"] = min(worst["section_min"], float(s.min()))
                worst["section_max"] = max(worst["section_max"], float(s.max()))
        for k, (a, b, jj, ww) in hero_w.items():
            vh = om.skin(gv[a:b], jj, ww, hero["bind"], ph, rh, sh)
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
    assert worst["section_min"] > 0.85 and worst["section_max"] < 1.02, \
        f"arm cross-section collapse/inflation beyond the thinning law ({worst['section_min']:.3f}..{worst['section_max']:.3f})"
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
        p, r, sc = om.pose_octopus(body, arms, "game", with_scale=True)
        vs = om.skin(gv, game["joints"], game["weights"], game["bind"], p, r, sc)
        ratio = tri_area(vs, tg) / rest_area
        shown["arm"] = min(shown["arm"], float(ratio[region == 0].min()))
        shown["crown"] = min(shown["crown"], float(ratio[region == 1].min()))
        shown["flips"] = max(shown["flips"], flips(vs, r))
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


def check_stretch(game, hero, report):
    """The stretch control (#106) at the SKIN: changing arm k's stretch moves
    only arm k's own tube and cup vertices -- every other arm, the crown, web,
    junction rings, eyes and siphon stay bit-identical (a stretch that leaks
    into a shared joint, the crown segments or a neighbour breaks it); and the
    free arm keeps its volume within 3 % while it elongates (radius ~
    1/sqrt(length factor); dropping the thinning, or thinning by 1/factor,
    breaks it)."""
    out = {}
    for d, name, trials in ((game, "game", 40), (hero, "hero", 8)):
        lod = d["lod"]
        for trial in range(trials):
            body, arms = random_body(RNG), [random_arm(RNG) for _ in range(8)]
            k = int(RNG.integers(8))
            moved = list(arms)
            moved[k] = om.ArmParams(**{**vars(arms[k]), "stretch": float((arms[k].stretch + 0.3 + 0.6 * RNG.random()) % 1.0)})
            v0 = om.skin(d["vertices"], d["joints"], d["weights"], d["bind"], *om.pose_octopus(body, arms, lod, with_scale=True))
            v1 = om.skin(d["vertices"], d["joints"], d["weights"], d["bind"], *om.pose_octopus(body, moved, lod, with_scale=True))
            changed = np.nonzero((v0 != v1).any(axis=1))[0]
            a, b = d["arm_ranges"][k]
            assert len(changed) > 0, "stretch moved nothing"
            assert changed.min() >= a + d["sectors"] and changed.max() < b, \
                f"{name}: arm {k}'s stretch moved a vertex outside its own free arm"
        out[f"{name}_trials"] = trials
    # volume while elongating, hero arm tube, straight arm
    sectors = hero["sectors"]
    rings = om.HERO_RINGS - 1 - om.JUNCTION_RING
    a = hero["arm_ranges"][0][0]
    vols = {}
    for st in (0.0, 0.5, 1.0):
        arms = [om.ArmParams(stretch=st)] + [om.ArmParams()] * 7
        v = om.skin(hero["vertices"], hero["joints"], hero["weights"], hero["bind"],
                    *om.pose_octopus(om.BodyParams(), arms, "hero", with_scale=True))
        rr = v[a:a + rings * sectors].reshape(rings, sectors, 3)
        area = np.linalg.norm(np.cross(rr, np.roll(rr, -1, axis=1)).sum(axis=1), axis=1) / 2
        c = rr.mean(axis=1)
        h = np.linalg.norm(np.diff(c, axis=0), axis=1)
        vols[st] = dict(volume_m3=float((0.5 * (area[1:] + area[:-1]) * h).sum()),
                        length_m=float(h.sum()), min_radius_ratio=None)
    for st in (0.5, 1.0):
        ratio = vols[st]["volume_m3"] / vols[0.0]["volume_m3"]
        vols[st]["volume_ratio"] = ratio
        assert abs(ratio - 1) < 0.03, f"stretch {st}: arm volume ratio {ratio:.3f}"
    out["volume_straight_arm"] = {str(k): {kk: vv for kk, vv in v.items() if vv is not None} for k, v in vols.items()}
    report["stretch"] = out


def check_sizes(report):
    """Every SIZES row builds a closed, bind-exact game mesh, and each 'mid' row
    spans 2.0 marines straight at rest and 2.5 fully stretched (Erik's sentence,
    #106). A row edited off those spans without re-solving its dials breaks it;
    a baby/boss row adds its own targets here."""
    targets = {"mid": (2.0, 2.5), "mid_bulk": (2.0, 2.5), "mid_heavy": (2.0, 2.5)}
    keep = om.SIZE
    out = {}
    try:
        for name in om.SIZES:
            om.set_size(name)
            r = om.size_report()
            if name in targets:
                lo, hi = targets[name]
                assert abs(r["span_rest_marines"] - lo) < 0.02, f"{name}: rest span {r['span_rest_marines']:.3f} marines"
                assert abs(r["span_stretched_marines"] - hi) < 0.02, f"{name}: stretched span {r['span_stretched_marines']:.3f}"
            if name != keep:
                d = om.assemble("game")
                mesh_checks(d, f"size_{name}_game", report)
            out[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    finally:
        om.set_size(keep)
    report["sizes"] = out


def main():
    report = {}
    t0 = time.time()
    report["active_size"] = om.SIZE
    check_controls(report)
    check_independence(report)
    check_continuity(report)
    game, hero = om.assemble("game"), om.assemble("hero")
    check_deformation.tg = mesh_checks(game, "game", report)
    mesh_checks(hero, "hero", report)
    check_deformation(game, hero, report)
    check_stretch(game, hero, report)
    check_sizes(report)
    report["seconds"] = round(time.time() - t0, 1)
    report["limits"] = ("Random poses span every control's full range including edges; no self-collision, "
                        "arm-arm or arm-body intersection or ground contact is checked "
                        "or solved. Crown/body areas are measured on the GAME skin; the hero shares its weights' "
                        "construction. Thresholds are prototype acceptance, not a production spec.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "property_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert not FAILS, FAILS
    print("OCTOPUS_PROPERTIES_OK")


if __name__ == "__main__":
    main()
