"""Properties from the approved brief, evaluated on actual weighted surfaces.

Standalone prototype checks, not the game test suite. No generated screenshots
or exact geometry counts are golden files. Run with this machine's Breach Python.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import arm_math as arm


def verify():
    """Protect geometry, length, weights, contact and shown bending quality.

    Broken winding/topology, unnormalised skin weights, scaling joint transforms,
    a discontinuous frame normal or over-tight curl must fail these properties.
    Thresholds describe this demonstration, not arbitrary future arm controls.
    """
    d = arm.geometry()
    v, j, w = d["vertices"], d["joints"], d["weights"]
    assert np.isfinite(v).all()
    assert np.min(w) >= 0 and np.max(np.abs(w.sum(axis=1) - 1)) < 1e-12
    edges = Counter()
    for face in d["faces"]:
        for a, b in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted([a, b]))] += 1
    assert all(count == 2 for count in edges.values()), "non-closed topology"
    assert np.max(np.abs(arm.skin(v, j, w, *arm.pose()) - v)) < 1e-12
    rest_rings = v[:(d["rings"] - 1) * d["sectors"]].reshape(-1, d["sectors"], 3)

    def areas(rings):
        # Oriented area vector magnitude: independent of the curve formula.
        return np.linalg.norm(np.cross(rings, np.roll(rings, -1, axis=1)).sum(axis=1), axis=1) / 2

    rest_areas = areas(rest_rings)
    minimum_area, maximum_area = 1., 1.
    min_triangle_ratio = 1.
    max_length_error = 0.
    min_cup_gap = 1.
    max_frame_step = 0.
    previous = None
    samples = []
    triangles = np.array([[f[0], f[k], f[k + 1]] for f in d["faces"] for k in range(1, len(f) - 1)])
    rest_tri_area = np.linalg.norm(np.cross(v[triangles[:, 1]] - v[triangles[:, 0]], v[triangles[:, 2]] - v[triangles[:, 0]]), axis=1)
    assert rest_tri_area.min() > 1e-12
    for t in np.linspace(0, arm.PERIOD, 181):
        control = arm.cycle_weights(t)
        points, frames = arm.pose(*control)
        p2, r2 = arm.pose(*control)
        assert np.array_equal(points, p2) and np.array_equal(frames, r2)
        assert np.isfinite(points).all() and np.isfinite(frames).all()
        assert np.max(np.abs(np.linalg.det(frames) - 1)) < 1e-10
        error = float(np.max(np.abs(np.linalg.norm(np.diff(points, axis=0), axis=1) - arm.STEP)))
        max_length_error = max(max_length_error, error)
        assert error < 1e-12
        if previous is not None:
            dots = (np.einsum("nij,nij->n", previous, frames) - 1) / 2
            step = float(np.arccos(np.clip(dots, -1, 1)).max())
            max_frame_step = max(max_frame_step, step)
            assert step < .19, "temporal frame jump at 20 Hz sample spacing"
        previous = frames
        bent = arm.skin(v, j, w, points, frames)
        rings = bent[:len(rest_rings) * d["sectors"]].reshape(rest_rings.shape)
        ratio = areas(rings) / rest_areas
        minimum_area, maximum_area = min(minimum_area, float(ratio.min())), max(maximum_area, float(ratio.max()))
        triangle_ratio = np.linalg.norm(np.cross(bent[triangles[:, 1]] - bent[triangles[:, 0]], bent[triangles[:, 2]] - bent[triangles[:, 0]]), axis=1) / rest_tri_area
        min_triangle_ratio = min(min_triangle_ratio, float(triangle_ratio.min()))
        assert ratio.min() > .97 and ratio.max() < 1.001, "cross-section collapse or inflation"
        assert triangle_ratio.min() > .52, "concave triangle collapsing"
        # Conservative sphere bounds around the ACTUAL deformed cup vertices.
        centres, radii = [], []
        for cup in d["cups"]:
            verts = bent[cup["start"]:cup["stop"]]
            c = verts.mean(axis=0)
            centres.append(c)
            radii.append(np.linalg.norm(verts - c, axis=1).max())
        centres, radii = np.array(centres), np.array(radii)
        gaps = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1) - radii[:, None] - radii[None, :]
        np.fill_diagonal(gaps, np.inf)
        min_cup_gap = min(min_cup_gap, float(gaps.min()))
        assert gaps.min() > 0, "cup bounding spheres overlap"
        samples.append(dict(seconds=float(t), area_min=float(ratio.min())))
    contact = arm.skin(v, j, w, *arm.pose(0, 1))
    tip = contact[d["tube_count"] - 2]
    contact_error = abs(float(tip[0] - arm.CONTACT_X))
    penetration = max(0., float(contact[:, 0].max() - arm.CONTACT_X))
    assert contact_error < 1e-8 and penetration < 1e-6
    # Continuity across the loop boundary, with all geometry following it.
    left = arm.skin(v, j, w, *arm.pose(*arm.cycle_weights(arm.PERIOD - 1e-6)))
    right = arm.skin(v, j, w, *arm.pose(*arm.cycle_weights(1e-6)))
    assert np.max(np.abs(left - right)) < 1e-8
    report = dict(pose_samples=len(samples), segment_length_max_error_m=max_length_error,
                  cross_section_area_ratio_min=minimum_area, cross_section_area_ratio_max=maximum_area,
                  triangle_area_ratio_min=min_triangle_ratio, cup_sphere_clearance_min_m=min_cup_gap,
                  temporal_frame_step_max_rad_at_20hz=max_frame_step,
                  contact_tip_error_m=contact_error, contact_surface_penetration_m=penetration,
                  contact_point_m=tip.tolist(),
                  limits="Checks cover 181 loop poses; 97% minimum section area and 52% minimum triangle area thresholds. Cup spheres prove cup/cup clearance only. No general body self-collision, wall collision, adhesion, arbitrary twist or grasping solver.")
    path = Path(__file__).resolve().parent.parent / "prototype" / "property_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
