"""Trunk rows from smooth edge profiles (numpy only; no Blender).

A half-body loft (`workwear.Figure`, `bodysuit.Suit`) is one row per landmark: (name, z, centre x,
centre y, half-width, half-depth front, back, exponent, level). Where the trunk becomes the leg the
centre swings sideways fast while the half-width changes the other way, and a handful of hand-set
rows there leave the OUTLINE -- what the eye judges -- as straight pieces meeting in corners: the
outer edge `cx + a` and the inner edge `cx - a` are each interpolated only through the rows.

`edge_rows` turns that round: the region is authored as the outlines themselves, as functions of
height, and the rows are generated from them densely (every `step` m), so the loft carries the
curves and nothing else:

  outer   the front view's outer edge (x of the left half's outline)
  inner   the front view's inner edge: the gap between the legs below the crotch; above it the
          (virtual) edge where the half-section reaches past the centre line -- where it crosses 0
          the legs meet. None = -outer (a section centred on the centre line)
  front   the side view's front-most y (negative: she faces -Y)
  back    the side view's back-most y
  cy      the section's centre y (where it is widest), between front and back
  n       the superellipse exponent

Each is a control table [(z, value), ...]; the rows' values come from a C2 cubic spline (`smooth`)
or a monotone cubic (`pchip`: never overshoots, for a curve that must not ripple, the gap), or
("pchip", z_a, z_b): monotone below z_a, C2 above z_b, blended between (the gap's edge, which
must not ripple at the crotch but must not carry a curvature step into the belly above). The
generated sections are LEVEL (horizontal) from `level_from` up, so there every row is a true
horizontal slice and the loft's outline at height z is the profiles' value; below it they turn
smoothly towards the leg's own sections (square to the centre line). The rows just outside the region
(`below`, `above`: the tables' own neighbouring rows) are added to the controls, so the curves leave
the region with the slopes those rows already have.
"""
import numpy as np


def natural_cubic(x, y, xq):
    """C2 natural cubic spline through (x, y), evaluated at xq (linear beyond the ends)."""
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n))
    r = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
        r[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, r)
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    hi = h[i]
    a, b = (x[i + 1] - xq) / hi, (xq - x[i]) / hi
    return a * y[i] + b * y[i + 1] + ((a ** 3 - a) * M[i] + (b ** 3 - b) * M[i + 1]) * hi * hi / 6.0


def pchip(x, y, xq):
    """Monotone piecewise cubic (Fritsch-Carlson): C1, no overshoot between the controls."""
    x, y, xq = np.asarray(x, float), np.asarray(y, float), np.asarray(xq, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(len(x))
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    t = (xq - x[i]) / h[i]
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * y[i] + (t ** 3 - 2 * t ** 2 + t) * h[i] * m[i]
            + (-2 * t ** 3 + 3 * t ** 2) * y[i + 1] + (t ** 3 - t ** 2) * h[i] * m[i + 1])


def _edges(row):
    """A table row -> its profile values (outer, inner, front, back, cy, n)."""
    _, z, cx, cy, a, bf, bb, n, _ = row
    return dict(outer=cx + a, inner=cx - a, front=cy - bf, back=cy + bb, cy=cy, n=n)


def edge_rows(spec, below=(), above=()):
    """Rows for z in [spec["z"][0], spec["z"][1]] every spec["step"] from the profile tables in
    `spec` (see the module doc); `below` / `above` = the neighbouring table rows (their values
    join the controls). spec["interp"] = {profile: "smooth" | "pchip"} (default smooth)."""
    z0, z1 = spec["z"]
    zs = np.round(np.arange(z0, z1 + 1e-9, spec["step"]), 6)
    ends = [_edges(r) for r in below] + [_edges(r) for r in above]
    ez = [r[1] for r in below] + [r[1] for r in above]
    interp = spec.get("interp", {})
    vals = {}
    for k in ("outer", "front", "back", "cy", "n", "inner"):
        ctrl = list(spec[k]) if spec.get(k) is not None else []
        if k == "inner" and spec.get("inner_meets_outer") is not None:
            # above this height the section is centred on the centre line: inner = -outer
            zm = spec["inner_meets_outer"]
            ctrl = [c for c in ctrl if c[0] < zm] + [(z, -float(vals["outer"][j])) for j, z in enumerate(zs) if z >= zm]
        ctrl += [(z, e[k]) for z, e in zip(ez, ends)]
        ctrl = sorted({round(z, 6): v for z, v in ctrl}.items())
        cz, cv = np.array([c[0] for c in ctrl]), np.array([c[1] for c in ctrl])
        how = interp.get(k, "smooth")
        if isinstance(how, tuple):  # ("pchip", z_a, z_b): monotone below z_a, C2 above z_b, blended between
            _, za, zb = how
            q = np.clip((zs - za) / (zb - za), 0.0, 1.0)
            q = q * q * (3.0 - 2.0 * q)
            vals[k] = (1.0 - q) * pchip(cz, cv, zs) + q * natural_cubic(cz, cv, zs)
        else:
            vals[k] = (pchip if how == "pchip" else natural_cubic)(cz, cv, zs)
    cxs = 0.5 * (vals["outer"] + vals["inner"])
    # the section planes: level from `level_from` up; below it they turn smoothly towards the
    # plane square to the centre line (the leg's own sections, which the rows below have), so
    # the tilt of the sections never jumps where the generated rows meet the table's
    lf = spec.get("level_from", z0)
    dcx, dcy = np.gradient(cxs, zs), np.gradient(vals["cy"], zs)
    rows = []
    for j, z in enumerate(zs):
        o, i = vals["outer"][j], vals["inner"][j]
        cx, a = 0.5 * (o + i), 0.5 * (o - i)
        cy = vals["cy"][j]
        level = 1
        if z < lf:
            s = (z - z0) / max(lf - z0, 1e-9)
            w = s * s * (3.0 - 2.0 * s)
            tg = np.array([dcx[j], dcy[j], 1.0])
            nv = w * np.array([0.0, 0.0, 1.0]) + (1.0 - w) * tg / np.linalg.norm(tg)
            level = tuple(float(v) for v in nv / np.linalg.norm(nv))
        rows.append(("%s_%03d" % (spec.get("name", "pelvis"), int(round(z * 1000))), float(z), float(cx), float(cy), float(a),
                     float(cy - vals["front"][j]), float(vals["back"][j] - cy), float(vals["n"][j]), level))
    return tuple(rows)
