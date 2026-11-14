"""Outline measurement of a built figure (numpy only; the Blender side is `mesh_outlines`).

An orthographic view's silhouette at height z is the extent of the figure's horizontal
section at z along the view's sideways axis. So the outlines are measured by SLICING the
mesh: every edge that crosses a level z gives a point of the section, and the outline is the
section's extent seen from each azimuth. No render, no pixel noise.

`turning(z, u)` is the outline's direction change per centimetre of height (degrees/cm): the
outline's angle to the vertical, fitted over a 1 cm window, differenced over 1 cm. A smooth
curve gives a low, slowly varying value; a corner shows as a spike.
"""
import math

import numpy as np

# the views measured: name -> azimuth (deg) of the view's sideways axis; the outline is the
# section's largest extent along (cos az, sin az) in the figure's x-y plane (faces -Y, +X her left)
VIEWS = (("view_front", 0.0), ("quarter_front", -45.0), ("quarter_back", 45.0))


def slice_points(V, E, z):
    """Points where the edges E (k, 2) of a mesh with vertices V cross the level z."""
    a, b = V[E[:, 0]], V[E[:, 1]]
    s = (a[:, 2] - z) * (b[:, 2] - z)
    m = s < 0.0
    a, b = a[m], b[m]
    w = (z - a[:, 2]) / (b[:, 2] - a[:, 2])
    return a + w[:, None] * (b - a)


def edges_of(faces):
    """The unique edges of a polygon list (each face a sequence of vertex indices)."""
    E = []
    for f in faces:
        f = list(f)
        E += [(f[i], f[(i + 1) % len(f)]) for i in range(len(f))]
    E = np.sort(np.asarray(E, np.int64), axis=1)
    return np.unique(E, axis=0)


def outlines(V, E, zs, x_min=0.0, gap_min=0.0015):
    """Per level z: the LEFT half's outer x (front and back views alike), the inner x of the gap
    between the legs (nan where the section is closed across the centre line), the front-most
    and back-most y (side view), and the extent along each `VIEWS` azimuth."""
    out = {k: np.full(len(zs), np.nan) for k in ("outer", "inner", "front", "back")}
    for name, _ in VIEWS:
        out[name] = np.full(len(zs), np.nan)
    for i, z in enumerate(zs):
        P = slice_points(V, E, z)
        if not len(P):
            continue
        L = P[P[:, 0] >= x_min - 1e-6]
        if len(L):
            out["outer"][i] = L[:, 0].max()
            xi = L[:, 0].min()
            out["inner"][i] = xi if xi > gap_min else np.nan
        out["front"][i], out["back"][i] = P[:, 1].min(), P[:, 1].max()
        for name, az in VIEWS:
            c, s = math.cos(math.radians(az)), math.sin(math.radians(az))
            out[name][i] = (P[:, 0] * c + P[:, 1] * s).max()
    return out


def turning(zs, u, window=0.010):
    """The outline u(z)'s direction change, degrees per cm, at each z (nan where undefined)."""
    zs, u = np.asarray(zs, float), np.asarray(u, float)
    h = 0.5 * window
    ang = np.full(len(zs), np.nan)
    for i, z in enumerate(zs):
        m = (np.abs(zs - z) <= h + 1e-9) & np.isfinite(u)
        if m.sum() >= 3:
            ang[i] = math.degrees(math.atan(np.polyfit(zs[m], u[m], 1)[0]))
    out = np.full(len(zs), np.nan)
    for i, z in enumerate(zs):
        j0, j1 = np.argmin(np.abs(zs - (z - h))), np.argmin(np.abs(zs - (z + h)))
        if np.isfinite(ang[j0]) and np.isfinite(ang[j1]):
            out[i] = abs(ang[j1] - ang[j0]) / (window * 100.0)
    return out


def report(zs, o, lo=0.60, hi=1.14, keys=("outer", "inner", "front", "back", "quarter_front", "quarter_back")):
    """{outline: (worst deg/cm, at z)} between lo and hi, and the turning arrays."""
    res, tr = {}, {}
    m = (zs >= lo) & (zs <= hi)
    for k in keys:
        t = turning(zs, o[k])
        tr[k] = t
        tt = np.where(m & np.isfinite(t), t, -1.0)
        j = int(np.argmax(tt))
        res[k] = (float(tt[j]), float(zs[j])) if tt[j] >= 0 else (float("nan"), float("nan"))
    return res, tr
