"""propgen — THE procedural prop / vegetation geometry source.

Pure numpy in, triangle arrays out. This module:

  * imports NOTHING from ``simulation`` and is never imported by it;
  * is seeded-deterministic — the same arguments always produce byte-identical
    arrays (``tests/test_propgen.py`` gates this);
  * is RENDER-ONLY float geometry (the Q16.16 iron rule's render exemption:
    nothing here ever reaches synced sim state, so plain float64/float32 math
    is correct and intended).

New flora = a generator function + a ``PALETTES`` row HERE, never inline mesh
code somewhere else (project CLAUDE.md canonical-systems rule, props &
vegetation arc #60 §7).

Public contract (FROZEN at promotion, arc #60 P2 — critique F27)::

    generate_tree(seed, palette, height, style, decor, decor_density)
    generate_palm(seed, palette, height)
        -> (verts f32 (N, 3), normals f32 (N, 3), colors u8 (N, 4)),  N % 3 == 0

``height`` stays an EXPLICIT parameter in authored units (metres) rather than
being normalized to 1.0 with a draw-time scale: tuft/frond counts are not
scale-stable (they are derived from absolute lobe size), so a normalized mesh
would change its own detail density when scaled. ``height`` therefore joins
any model cache key — see ``renderer/static_props.py``.

**Vertex-color ALPHA is DATA, never opacity**: it carries the wind-flutter
weight (0 = rigid trunk … 1 = leaf tuft) that the P4 sway vertex shader reads.
Every consumer must force its output alpha to 1.0.

Style target (Erik's reference, 2026-09-06): smooth rounded canopy lobes with
bright leaf-cluster detail over a darker mass. Two styles:
  "smooth"  — ellipsoid normals, per-lobe + canopy-wide gradient, leaf tufts
  "faceted" — flat-shaded low-poly facets (the Kenney-ish look)

Decor (blossoms / fruit) is emitted in CLUSTERS, not uniform scatter: under
the game's straight-down orthographic camera a uniform scatter reads as
confetti (P0 top-down finding, 2026-09-07). Fewer, larger, clumped elements
read as deliberate blossom/fruit masses from above.

Promoted 2026-09 from ``prototypes/prop_spike/treegen.py`` (spike blessed by
Erik in perspective + top-down). Recursive branching is textbook; no published
technique beyond that. See
``docs/architecture/graphics/props_and_vegetation.md`` §2 / §4.3 / §7.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Palettes — HSV ranges (h in degrees). A new look is a ROW here.
# ---------------------------------------------------------------------------

PALETTES: dict[str, dict[str, tuple[float, float]]] = {
    "green": {
        "leaf_h": (100.0, 128.0), "leaf_s": (0.62, 0.78), "leaf_v": (0.82, 0.95),
        "bark_h": (18.0, 30.0),   "bark_s": (0.50, 0.62), "bark_v": (0.26, 0.36),
        "flower_h": (315.0, 355.0), "flower_s": (0.10, 0.42), "flower_v": (0.95, 1.0),
        "fruit_h": (0.0, 22.0),     "fruit_s": (0.75, 0.90), "fruit_v": (0.68, 0.85),
    },
    "autumn": {
        "leaf_h": (18.0, 55.0),  "leaf_s": (0.60, 0.85), "leaf_v": (0.65, 0.88),
        "bark_h": (18.0, 28.0),  "bark_s": (0.45, 0.60), "bark_v": (0.26, 0.36),
        "flower_h": (40.0, 55.0),  "flower_s": (0.25, 0.50), "flower_v": (0.95, 1.0),
        "fruit_h": (25.0, 40.0),   "fruit_s": (0.80, 0.92), "fruit_v": (0.70, 0.85),
    },
    "exotic": {
        "leaf_h": (255.0, 320.0), "leaf_s": (0.42, 0.62), "leaf_v": (0.65, 0.90),
        "bark_h": (200.0, 230.0), "bark_s": (0.15, 0.30), "bark_v": (0.24, 0.34),
        "flower_h": (160.0, 200.0), "flower_s": (0.55, 0.85), "flower_v": (0.92, 1.0),
        "fruit_h": (45.0, 62.0),    "fruit_s": (0.80, 0.95), "fruit_v": (0.80, 0.95),
    },
}

# Valid values for the frozen signature's enum-ish parameters (the P3 entity
# registry validates its `style` / `decor` / `palette` choices against these).
STYLES: tuple[str, ...] = ("smooth", "faceted")
DECORS: tuple[str, ...] = ("flowers", "fruit")


# ---------------------------------------------------------------------------
# Small vector / colour helpers
# ---------------------------------------------------------------------------

def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """HSV (h in degrees, s/v in [0, 1]) -> float32 RGB in [0, 1]."""
    h = (h % 360.0) / 60.0
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return np.array([r, g, b], dtype=np.float32)


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3-vector cross product. Numerically identical to ``np.cross`` (same
    products, same order) but ~5x cheaper: ``np.cross`` pays for its
    moveaxis/broadcast machinery on every call, and this generator calls it
    tens of thousands of times per tree (measured: ~45% of generation time)."""
    return np.array((a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]))


def _norm3(a: np.ndarray) -> float:
    """Euclidean length of a 3-vector (cheaper than ``np.linalg.norm``)."""
    return math.sqrt(float(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))


def _perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to (unit) *axis*."""
    a = _UNIT_X if abs(axis[1]) > 0.9 else _UNIT_Y
    u = _cross3(axis, a)
    u /= _norm3(u)
    return u, _cross3(axis, u)


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of *v* about the unit *axis* by *ang* radians."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + _cross3(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _rand_unit(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-distributed unit vector on the sphere."""
    d = rng.normal(size=3)
    return d / max(_norm3(d), 1e-9)


_UNIT_X = np.array([1.0, 0.0, 0.0])
_UNIT_Y = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Icosphere — the canopy blob primitive.
# ---------------------------------------------------------------------------

def _icosphere(n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere after *n_sub* subdivisions: (verts (V,3), faces (F,3))."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    vlist = [v for v in verts]
    for _ in range(n_sub):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = (vlist[i] + vlist[j]) / 2.0
                m /= np.linalg.norm(m)
                cache[key] = len(vlist)
                vlist.append(m)
            return cache[key]

        nxt = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nxt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nxt
    return np.array(vlist), np.array(faces, dtype=np.int32)


_ICO0_V, _ICO0_F = _icosphere(0)   # fruit / coconuts       (12 v,  20 f)
_ICO1_V, _ICO1_F = _icosphere(1)   # faceted canopy blobs   (42 v,  80 f)
_ICO2_V, _ICO2_F = _icosphere(2)   # smooth canopy blobs   (162 v, 320 f)


# ---------------------------------------------------------------------------
# Triangle-array accumulator (non-indexed; each emitter supplies its normals).
# ---------------------------------------------------------------------------

class _Soup:
    """Triangle soup accumulator.

    Vertex-color ALPHA carries the wind-flutter weight (0 = rigid trunk,
    1 = leaf tuft) — the P4 sway shader reads it; it is NEVER transparency.
    """

    __slots__ = ("verts", "normals", "colors", "flut")

    def __init__(self) -> None:
        self.verts: list[np.ndarray] = []
        self.normals: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.flut: list[float] = []

    def tri(self, pts, norms, cols, flut: float = 0.0) -> None:
        self.verts += list(pts)
        self.normals += list(norms)
        self.colors += list(cols)
        self.flut += [flut] * 3

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = np.asarray(self.verts, dtype=np.float32)
        n = np.asarray(self.normals, dtype=np.float32)
        cols = np.clip(np.asarray(self.colors, dtype=np.float32), 0.0, 1.0)
        a = (np.asarray(self.flut, dtype=np.float32)[:, None] * 255).astype(np.uint8)
        rgba = np.concatenate([(cols * 255).astype(np.uint8), a], axis=1)
        return v, n, rgba


def _face_normal(a, b, c) -> np.ndarray:
    fn = _cross3(b - a, c - a)
    return fn / max(_norm3(fn), 1e-12)


def _emit_tube(soup: _Soup, p0, p1, r0, r1, col_lo, col_hi,
               sides: int = 8, smooth: bool = True) -> None:
    """A tapered cylinder from *p0* (radius r0) to *p1* (radius r1)."""
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return
    axis = axis / length
    u, w = _perp_basis(axis)
    ang = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w)
    lo = p0 + ring * r0
    hi = p1 + ring * r1
    for i in range(sides):
        j = (i + 1) % sides
        tri1 = (lo[i], lo[j], hi[j])
        tri2 = (lo[i], hi[j], hi[i])
        if smooth:
            soup.tri(tri1, (ring[i], ring[j], ring[j]),
                     (col_lo, col_lo, col_hi))
            soup.tri(tri2, (ring[i], ring[j], ring[i]),
                     (col_lo, col_hi, col_hi))
        else:
            for tri in (tri1, tri2):
                fn = _face_normal(*tri)
                soup.tri(tri, (fn, fn, fn), (col_lo, col_lo, col_hi))


def _emit_blob(soup: _Soup, rng: np.random.Generator, center: np.ndarray,
               size: float, hsv: tuple, canopy_y: tuple[float, float],
               smooth: bool) -> None:
    """One canopy blob.

    Brightness follows a per-LOBE gradient blended with the CANOPY-WIDE one
    (bright crown, dark underside) — that gradient, more than the geometry, is
    most of the reference look. The "smooth" style then scatters small bright
    tangent quads ("tufts") over the surface for the fluffy leaf-cluster read.
    """
    ico_v, ico_f = (_ICO2_V, _ICO2_F) if smooth else (_ICO1_V, _ICO1_F)
    scale = np.array([size, size * rng.uniform(0.78, 0.92), size])
    jit = 0.045 if smooth else 0.10
    jitter = 1.0 + rng.normal(0.0, jit, size=(len(ico_v), 1))
    pts = ico_v * jitter * scale + center
    y_lo, y_hi = canopy_y
    blob_lo = center[1] - scale[1]
    blob_hi = center[1] + scale[1]
    h, s, v = hsv

    def vert_color(p: np.ndarray, flick: float) -> np.ndarray:
        rc = np.clip((p[1] - y_lo) / max(y_hi - y_lo, 1e-9), 0.0, 1.0)
        rb = np.clip((p[1] - blob_lo) / max(blob_hi - blob_lo, 1e-9), 0.0, 1.0)
        rel = 0.45 * rc + 0.55 * rb
        vv = v * (0.36 + 0.82 * rel ** 1.3)
        ss = s * (1.05 - 0.22 * rel)
        return _hsv_to_rgb(h + flick, min(ss, 1.0), min(vv, 1.0))

    if not smooth:
        for f in ico_f:
            tri = pts[f]
            fn = _face_normal(*tri)
            c = vert_color(tri.mean(axis=0), rng.uniform(-5.0, 5.0))
            soup.tri(tri, (fn, fn, fn), (c, c, c), flut=0.30)
        return

    # Smooth: ellipsoid normal n ∝ (p - c) / scale².
    nrm = (pts - center) / (scale ** 2)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    vcols = [vert_color(p, 0.0) * 0.90 for p in pts]  # darker core mass
    for f in ico_f:
        soup.tri(pts[f], nrm[f], [vcols[f[0]], vcols[f[1]], vcols[f[2]]],
                 flut=0.30)

    # Leaf tufts: small bright tangent quads on the surface.
    n_tufts = int(_TUFTS_PER_UNIT_BLOB * (size / _BLOB_REF_SIZE) ** 2)
    for _ in range(n_tufts):
        d = _rand_unit(rng)
        if d[1] < -0.25 and rng.uniform() < 0.65:
            continue  # sparse underside
        p = center + d * scale * (1.0 + rng.uniform(-0.02, 0.07))
        t1, t2 = _perp_basis(d)
        rot = rng.uniform(0.0, 2.0 * math.pi)
        a = math.cos(rot) * t1 + math.sin(rot) * t2
        b = -math.sin(rot) * t1 + math.cos(rot) * t2
        ts = size * rng.uniform(0.075, 0.135)
        lift = d * ts * 0.30
        p0, p2 = p + a * ts, p - a * ts
        p1, p3 = p + b * ts * 0.7 + lift, p - b * ts * 0.7 + lift
        c = np.clip(vert_color(p, rng.uniform(-8.0, 8.0))
                    * rng.uniform(1.02, 1.22), 0.0, 1.0)
        soup.tri((p0, p1, p2), (d, d, d), (c, c, c), flut=1.0)
        soup.tri((p0, p2, p3), (d, d, d), (c, c, c), flut=1.0)


# Tuft density: this many tufts on a blob of radius _BLOB_REF_SIZE, scaling
# with surface area. (Absolute, NOT height-relative — the reason `height`
# cannot be normalized away; see the module docstring.)
_TUFTS_PER_UNIT_BLOB = 240
_BLOB_REF_SIZE = 0.45


# ---------------------------------------------------------------------------
# Decor — CLUSTERED blossoms / fruit (P0 top-down finding, 2026-09-07)
# ---------------------------------------------------------------------------
# A uniform scatter of small elements reads as confetti under the game's
# straight-down ortho camera: individually sub-pixel-ish specks spread evenly
# over the canopy carry no shape. Clumping the same visual budget into a few
# larger masses gives the top-down silhouette something to read. Numbers below
# are the P2 feel pass (verified by top-down + perspective renders).

_DECOR_SPEC = {
    # kind:    (clusters_lo, clusters_hi, per_lo, per_hi, spread, size_lo, size_hi)
    "flowers": (0, 3, 4, 7, 0.30, 0.160, 0.240),
    "fruit":   (0, 3, 3, 5, 0.26, 0.120, 0.160),
}
# The cluster count is drawn per CANOPY BLOB and a tree has ~8-27 of them, so
# the low end is deliberately 0: most blobs carry nothing and the few that do
# carry one readable mass. Uniform per-blob decor is exactly the confetti the
# P0 top-down render rejected.


def _emit_decor(soup: _Soup, rng: np.random.Generator, center: np.ndarray,
                scale: np.ndarray, size: float, kind: str, pal: dict,
                density: float, others=()) -> None:
    """Blossom / fruit CLUSTERS on one canopy blob's surface.

    flowers: bright tangent quads (like larger pale tufts), upper-biased.
    fruit:   small hanging faceted spheres, outer/lower-biased.

    Each cluster picks one surface direction, then places 2-6 elements inside a
    small angular neighbourhood of it, so the result reads as a few deliberate
    masses rather than an even sprinkle (see the section note above).
    """
    if kind not in _DECOR_SPEC or density <= 0.0:
        return
    c_lo, c_hi, p_lo, p_hi, spread, s_lo, s_hi = _DECOR_SPEC[kind]
    # Density scales the CLUSTER COUNT (never the element size — the size is
    # what makes a clump readable from above).
    n_clusters = int(round(rng.integers(c_lo, c_hi) * density))
    if n_clusters <= 0:
        return
    key = "flower" if kind == "flowers" else "fruit"
    h = rng.uniform(*pal[key + "_h"])
    s = rng.uniform(*pal[key + "_s"])
    v = rng.uniform(*pal[key + "_v"])

    for _ in range(n_clusters):
        # --- cluster anchor: one surface direction, biased per kind ---------
        anchor = None
        for _try in range(8):
            d = _rand_unit(rng)
            if kind == "flowers" and d[1] < 0.0:
                continue          # blossoms live on the sunny top
            if kind == "fruit" and d[1] > 0.60:
                continue          # fruit hangs off the outer canopy, not the
                                  # crown apex — but high enough that the
                                  # straight-down camera still sees it (a
                                  # strictly lower-biased fruit tree reads
                                  # bare from above; P0 top-down finding)
            p = center + d * scale * 1.05
            # Skip anchors swallowed by a neighbouring blob — invisible anyway.
            if any(np.sum(((p - c2) / s2) ** 2) < 0.95 for c2, s2 in others):
                continue
            anchor = d
            break
        if anchor is None:
            continue

        t1, t2 = _perp_basis(anchor)
        n_elems = int(rng.integers(p_lo, p_hi + 1))
        for _e in range(n_elems):
            # Jitter the direction inside a small tangent disc around the
            # anchor, then re-normalize back onto the blob surface.
            off = rng.normal(0.0, spread * 0.5, size=2)
            d = anchor + t1 * off[0] + t2 * off[1]
            d /= max(_norm3(d), 1e-9)
            # Fruit sits PROUD of the leaf surface (an object ON the canopy,
            # not a decal in it) — flush with the surface it vanishes into the
            # tuft layer under the straight-down camera. Blossoms hug it.
            radial = (1.12 if kind == "fruit" else 1.0) + rng.uniform(0.02, 0.08)
            p = center + d * scale * radial
            if any(np.sum(((p - c2) / s2) ** 2) < 0.95 for c2, s2 in others):
                continue
            col = _hsv_to_rgb(h + rng.uniform(-6.0, 6.0),
                              s * rng.uniform(0.85, 1.1), v)
            if kind == "flowers":
                e1, e2 = _perp_basis(d)
                rot = rng.uniform(0.0, 2.0 * math.pi)
                a = math.cos(rot) * e1 + math.sin(rot) * e2
                b = -math.sin(rot) * e1 + math.cos(rot) * e2
                ts = size * rng.uniform(s_lo, s_hi)
                p0, p2 = p + a * ts, p - a * ts
                p1, p3 = p + b * ts + d * ts * 0.4, p - b * ts + d * ts * 0.4
                for x, y, z in ((p0, p1, p2), (p0, p2, p3)):
                    soup.tri((x, y, z), (d, d, d), (col, col, col), flut=1.0)
                    soup.tri((x, z, y), (-d, -d, -d), (col, col, col), flut=1.0)
            else:
                # A 20-face icosahedron: the fruit are small and CLUMPED, so
                # the coarser primitive is indistinguishable on screen and
                # keeps a fruiting tree from doubling the whole tri budget.
                r = size * rng.uniform(s_lo, s_hi)
                pts = _ICO0_V * r + p - np.array([0.0, r * 0.45, 0.0])  # hangs
                dark = col * 0.72
                for f in _ICO0_F:
                    tri = pts[f]
                    fn = _face_normal(*tri)
                    cc = col if tri[:, 1].mean() > p[1] - r * 0.8 else dark
                    soup.tri(tri, (fn, fn, fn), (cc, cc, cc), flut=0.55)


# ---------------------------------------------------------------------------
# Generators — the FROZEN public surface
# ---------------------------------------------------------------------------

def generate_tree(seed: int, palette: str = "green", height: float = 2.2,
                  style: str = "smooth", decor: str | None = None,
                  decor_density: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One broadleaf tree as triangle arrays.

    Args:
        seed: any int; the ONLY source of randomness. Same args -> same mesh.
        palette: a key of :data:`PALETTES`.
        height: authored height in metres (the trunk base sits at y = 0 and the
            canopy tops out near ``height``). NOT normalized — see the module
            docstring; it joins the model cache key.
        style: "smooth" (reference look) or "faceted".
        decor: None / "" for bare, else "flowers" or "fruit".
        decor_density: multiplier on the number of decor CLUSTERS (0 = none).

    Returns:
        ``(verts, normals, colors)`` — verts/normals float32 ``(N, 3)``,
        colors uint8 ``(N, 4)`` with ALPHA = flutter weight (NOT opacity),
        ``N % 3 == 0``, triangles in CCW winding.
    """
    rng = np.random.default_rng(seed)
    pal = PALETTES[palette]
    smooth = style == "smooth"
    soup = _Soup()

    bark_h = rng.uniform(*pal["bark_h"])
    bark_s = rng.uniform(*pal["bark_s"])
    bark_v = rng.uniform(*pal["bark_v"])
    bark_lo = _hsv_to_rgb(bark_h, bark_s, bark_v * 0.75)
    bark_hi = _hsv_to_rgb(bark_h, bark_s, bark_v * 1.20)

    tips: list[tuple[np.ndarray, float]] = []
    segs: list[tuple[np.ndarray, np.ndarray, float, float]] = []

    def branch(pos, direction, length, radius, depth) -> None:
        end = pos + direction * length
        segs.append((pos, end, radius, radius * 0.60))
        if depth == 0:
            tips.append((end, length))
            return
        n_children = int(rng.integers(2, 4))
        u, w = _perp_basis(direction)
        azim0 = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(n_children):
            azim = azim0 + k * (2.0 * math.pi / n_children) + rng.uniform(-0.4, 0.4)
            side = u * math.cos(azim) + w * math.sin(azim)
            rot_axis = _cross3(direction, side)
            rot_axis /= max(_norm3(rot_axis), 1e-9)
            tilt = rng.uniform(math.radians(24), math.radians(44))
            child_dir = _rotate(direction, rot_axis, tilt)
            child_dir /= np.linalg.norm(child_dir)
            branch(end, child_dir, length * rng.uniform(0.60, 0.74),
                   radius * 0.62, depth - 1)

    trunk_len = height * rng.uniform(0.40, 0.50)
    trunk_dir = np.array([rng.uniform(-0.07, 0.07), 1.0, rng.uniform(-0.07, 0.07)])
    trunk_dir /= np.linalg.norm(trunk_dir)
    branch(np.array([0.0, 0.0, 0.0]), trunk_dir, trunk_len,
           height * rng.uniform(0.042, 0.055), 3)

    for p0, p1, r0, r1 in segs:
        _emit_tube(soup, p0, p1, r0, r1, bark_lo, bark_hi, smooth=smooth)

    # --- canopy: a few LARGE overlapped blobs + one unifying crown blob ---
    blob_scale = rng.uniform(0.80, 1.00)
    blobs: list[tuple[np.ndarray, float]] = []
    for tip_pos, tip_len in tips:
        size = max(tip_len * blob_scale, height * 0.20)
        blobs.append((tip_pos + np.array([0.0, tip_len * 0.10, 0.0]), size))
    crown_c = np.mean([b[0] for b in blobs], axis=0)
    crown_s = max(b[1] for b in blobs) * rng.uniform(1.15, 1.35)
    blobs.append((crown_c + np.array([0.0, crown_s * 0.25, 0.0]), crown_s))

    ys = [c[1] for c, s in blobs]
    ss = [s for c, s in blobs]
    canopy_y = (min(y - s * 0.9 for y, s in zip(ys, ss)),
                max(y + s * 0.9 for y, s in zip(ys, ss)))

    leaf_h = rng.uniform(*pal["leaf_h"])
    leaf_s = rng.uniform(*pal["leaf_s"])
    leaf_v = rng.uniform(*pal["leaf_v"])
    ell = [(c, np.array([s, s * 0.85, s])) for c, s in blobs]
    want_decor = bool(decor)
    for i, (c, s) in enumerate(blobs):
        hsv = (leaf_h + rng.uniform(-5.0, 5.0), leaf_s, leaf_v)
        _emit_blob(soup, rng, c, s, hsv, canopy_y, smooth)
        if want_decor:
            others = [e for j, e in enumerate(ell) if j != i]
            _emit_decor(soup, rng, c, ell[i][1], s, decor, pal,
                        decor_density, others)

    return soup.arrays()


def generate_palm(seed: int, palette: str = "green", height: float = 2.8
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stylized palm: leaning curved trunk + a crown of serrated, drooping
    ribbon fronds (double-sided) + coconuts.

    Same triangle-array contract as :func:`generate_tree`; ``height`` is
    likewise authored metres and part of the cache key.
    """
    rng = np.random.default_rng(seed)
    pal = PALETTES[palette]
    soup = _Soup()
    up = np.array([0.0, 1.0, 0.0])

    bark_h = rng.uniform(*pal["bark_h"])
    bark_s = rng.uniform(*pal["bark_s"]) * 0.85
    bark_v = rng.uniform(*pal["bark_v"]) * 1.35
    bark_lo = _hsv_to_rgb(bark_h, bark_s, bark_v * 0.80)
    bark_hi = _hsv_to_rgb(bark_h, bark_s, bark_v * 1.15)

    # --- trunk: chain of segments bending around one fixed horizontal axis ---
    n_seg = 6
    lean_az = rng.uniform(0.0, 2.0 * math.pi)
    bend_axis = np.array([math.cos(lean_az), 0.0, math.sin(lean_az)])
    direction = np.array([0.0, 1.0, 0.0])
    direction = _rotate(direction, bend_axis, math.radians(rng.uniform(2.0, 10.0)))
    pos = np.array([0.0, 0.0, 0.0])
    seg_len = height * 0.82 / n_seg
    r_base, r_top = height * 0.040, height * 0.024
    bend_per_seg = math.radians(rng.uniform(2.0, 6.0))
    for i in range(n_seg):
        r0 = r_base + (r_top - r_base) * i / n_seg
        r1 = r_base + (r_top - r_base) * (i + 1) / n_seg
        end = pos + direction * seg_len
        _emit_tube(soup, pos, end, r0, r1, bark_lo, bark_hi, smooth=True)
        direction = _rotate(direction, bend_axis, bend_per_seg)
        direction /= np.linalg.norm(direction)
        pos = end
    top = pos

    # --- fronds ---
    leaf_h = rng.uniform(*pal["leaf_h"])
    leaf_s = rng.uniform(*pal["leaf_s"])
    leaf_v = rng.uniform(*pal["leaf_v"])
    n_fronds = int(rng.integers(15, 21))
    steps = 8
    for k in range(n_fronds):
        azim = 2.0 * math.pi * k / n_fronds + rng.uniform(-0.15, 0.15)
        horiz = np.array([math.cos(azim), 0.0, math.sin(azim)])
        theta0 = math.radians(rng.uniform(28.0, 62.0))
        d = horiz * math.cos(theta0) + up * math.sin(theta0)
        length = height * rng.uniform(0.42, 0.58)
        seg = length / steps
        droop = rng.uniform(0.16, 0.26)
        w_max = length * 0.15
        h_k = leaf_h + rng.uniform(-7.0, 7.0)
        p = top.copy()
        prev = None  # (left, right, color)
        for s in range(steps + 1):
            t = s / steps
            serr = 1.0 if s % 2 == 0 else 0.5     # zig-zag edge = leaflets
            w = w_max * math.sin(min(t * 1.15, 1.0) * math.pi) * serr
            side = _cross3(d, up)
            n_side = _norm3(side)
            side = (side / n_side if n_side > 1e-6 else _perp_basis(d)[0])
            left, right = p - side * w, p + side * w
            col = _hsv_to_rgb(h_k, leaf_s * (1.05 - 0.2 * t),
                              leaf_v * (0.62 + 0.55 * t))
            if prev is not None:
                pl, pr, pc = prev
                f_t = 0.35 + 0.6 * t   # fronds flutter more toward the tip
                for a, b, c, ca, cb, cc in (
                        (pl, pr, right, pc, pc, col),
                        (pl, right, left, pc, col, col)):
                    fn = _face_normal(a, b, c)
                    soup.tri((a, b, c), (fn, fn, fn), (ca, cb, cc), flut=f_t)
                    soup.tri((a, c, b), (-fn, -fn, -fn), (ca, cc, cb), flut=f_t)
            prev = (left, right, col)
            d = d - up * droop * (0.5 + t)
            d /= np.linalg.norm(d)
            p = p + d * seg

    # --- coconuts ---
    nut_col = _hsv_to_rgb(bark_h, bark_s * 1.1, bark_v * 0.55)
    for _ in range(int(rng.integers(2, 5))):
        az = rng.uniform(0.0, 2.0 * math.pi)
        c = top + np.array([math.cos(az), 0.0, math.sin(az)]) * height * 0.035 \
            - up * height * 0.045
        r = height * rng.uniform(0.030, 0.040)
        pts = _ICO0_V * r + c
        for f in _ICO0_F:
            fn = _face_normal(*pts[f])
            soup.tri(pts[f], (fn, fn, fn), (nut_col, nut_col, nut_col))

    return soup.arrays()


# Registry of the generators by the name the (P3) prop entity stores.
GENERATORS = {
    "tree": generate_tree,
    "palm": generate_palm,
}


__all__ = ["generate_tree", "generate_palm", "GENERATORS", "PALETTES",
           "STYLES", "DECORS"]


if __name__ == "__main__":  # pragma: no cover - a hand-run budget probe
    import time

    for label, fn, kw in (("tree smooth", generate_tree, {}),
                          ("tree faceted", generate_tree, {"style": "faceted"}),
                          ("tree flowers", generate_tree, {"decor": "flowers"}),
                          ("tree fruit", generate_tree, {"decor": "fruit"}),
                          ("palm", generate_palm, {})):
        t0 = time.perf_counter()
        v, n, c = fn(1, **kw)
        ms = (time.perf_counter() - t0) * 1000.0
        nbytes = v.nbytes + n.nbytes + c.nbytes
        print(f"{label:14s} {len(v) // 3:6d} tris  {ms:7.1f} ms  "
              f"{nbytes / 1024.0:8.1f} KiB  "
              f"y in [{v[:, 1].min():.2f}, {v[:, 1].max():.2f}]")
