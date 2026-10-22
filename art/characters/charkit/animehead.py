"""A stylised (anime-style 3D) head as ONE implicit surface, plus its eyes, lashes, ears and a hair blockout.

Beatrice B2a ("the face test"); written so a later anime-style character is a set of tables on it. Everything is
in metres in the world frame (x her left, -y the front, z up), symmetric about x = 0. The tables are the
character's (`FACE`, `EYES`, `HAIR` in her `beatrice.py`); nothing here holds a colour or a measure of hers.

The skin is a signed field, negative inside, every union and cut C2 (`smin3`, the same cubic as
`bodysuit.smin3`, which imports bpy):

  egg     a radial shape about `centre`: per elevation beta the drawn profile's front and back radii and the
          front view's side radius (each a natural cubic through hand-picked outline points, polar about the
          centre), blended round the head by superellipse exponents (front and back halves)
  neck    superellipse rings, natural cubic in z, capped inside the collar and inside the head; C2-united
  forms   small SDF primitives united C2 (nose, lips) and Gaussian displacements (brow, socket, cheeks, chin)
  cuts    the mouth line (a thin slot) and the eye openings: a prism along the view axis over the drawn almond
          (the front view's lid curves), C2-rounded into the lid margin, only within `depth` of the egg

It is meshed by surface nets on a banded grid (only blocks near the surface are sampled finely), projected onto
the field and shaded with the field's own normals. Separate closed objects: the eye surfaces (the uncut skin's
own surface set back behind the opening, its iris painted along the view axis, so it reads as drawn from the front
and foreshortens as the head turns), the upper lash bands (swept), the ears (`parts.ear` on an adapter) and the
hair (a cap following the skull and swept locks that fall clear of the head and the suit).
"""
import math

import numpy as np


# ------------------------------------------------------------------------------------------------ small kit
def smin3(a, b, k):
    """C2 smooth minimum (cubic), as `bodysuit.smin3`."""
    if np.ndim(k) == 0 and k <= 0:
        return np.minimum(a, b)
    k = np.maximum(k, 1e-9)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h ** 3 * k / 6.0


def smax3(a, b, k):
    return -smin3(-a, -b, k)


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class Curve:
    """A natural cubic through (x, y) points (sorted by x), linear beyond its ends."""

    def __init__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        n = len(x)
        h = np.diff(x)
        A, r = np.zeros((n, n)), np.zeros(n)
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2.0 * (h[i - 1] + h[i]), h[i]
            r[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        self.x, self.y, self.h, self.M = x, y, h, np.linalg.solve(A, r)
        M = self.M
        self.s0 = (y[1] - y[0]) / h[0] - h[0] * (2 * M[0] + M[1]) / 6.0
        self.s1 = (y[-1] - y[-2]) / h[-1] + h[-1] * (M[-2] + 2 * M[-1]) / 6.0

    def __call__(self, q):
        q = np.asarray(q, float)
        x, y, h, M = self.x, self.y, self.h, self.M
        qc = np.clip(q, x[0], x[-1])
        i = np.clip(np.searchsorted(x, qc, side="right") - 1, 0, len(x) - 2)
        hi, t, a = h[i], qc - x[i], x[i + 1] - qc
        v = (M[i] * a ** 3 + M[i + 1] * t ** 3) / (6 * hi) + (y[i] / hi - M[i] * hi / 6) * a + (y[i + 1] / hi - M[i + 1] * hi / 6) * t
        v = np.where(q < x[0], y[0] + self.s0 * (q - x[0]), v)
        return np.where(q > x[-1], y[-1] + self.s1 * (q - x[-1]), v)

    def d(self, q, e=1e-5):
        return (self(q + e) - self(q - e)) / (2 * e)


def ellipsoid(P, c, r, axes=None):
    """Approximate signed distance to an ellipsoid (centre c, radii r along `axes` rows, default the world axes)."""
    q = P - np.asarray(c, float)
    if axes is not None:
        q = q @ np.asarray(axes, float).T
    r = np.asarray(r, float)
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (r * r), axis=-1)
    return np.where(k1 > 1e-12, k0 * (k0 - 1.0) / np.maximum(k1, 1e-12), -r.min())


def capsule(P, a, b, ra, rb, ratio=1.0, hint=(0.0, -1.0, 0.0)):
    """Tapered capsule a -> b (radius ra -> rb across, ratio x that along `hint`), rounded ends; approximate SDF."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = unit(b - a)
    L = float(np.linalg.norm(b - a))
    V = unit(np.asarray(hint, float) - (np.asarray(hint, float) @ T) * T)
    U = np.cross(V, T)
    v = P - a
    w = v @ T
    s = np.clip(w / L, 0.0, 1.0)
    q = v - (s * L)[:, None] * T
    r = ra + (rb - ra) * s
    x, y, z = q @ U, q @ V, q @ T
    return (np.sqrt((x / r) ** 2 + (y / (r * ratio)) ** 2 + (z / r) ** 2) - 1.0) * r * min(1.0, ratio)


def mirror_x(P):
    Q = np.array(P, float)
    Q[:, 0] = np.abs(Q[:, 0])
    return Q


# ------------------------------------------------------------------------------------------------ the skin
class Egg:
    """The radial shape: |P - c| - rho(direction) (see the module docstring)."""

    def __init__(self, sp):
        self.c = np.asarray(sp["centre"], float)
        cy, cz = self.c[1], self.c[2]

        def polar(pts, h_of):
            pts = [(h_of(p), p[1]) for p in pts]
            b = np.array([math.atan2(z - cz, h) for h, z in pts])
            r = np.array([math.hypot(h, z - cz) for h, z in pts])
            return Curve(b, r)

        self.rf = polar(sp["front"], lambda p: cy - p[0])     # (y, z) points, the drawn profile's front
        self.rb = polar(sp["back"], lambda p: p[0] - cy)      # (y, z), its back
        self.rs = polar(sp["side"], lambda p: p[0])           # (x, z), the front view's side
        nf, nb = zip(*sp["n_front"]), zip(*sp["n_back"])
        nf, nb = [np.asarray(v, float) for v in nf], [np.asarray(v, float) for v in nb]
        self.nf, self.nb = Curve(np.radians(nf[0]), nf[1]), Curve(np.radians(nb[0]), nb[1])
        # B2b: `front_smooth` = (sigma along the elevation, sigma round the head, degrees): away from the midline the
        # profile's front radius is a SMOOTHED copy (a Gaussian along the elevation), blended in round the head, so the
        # profile's own dents (under the nose, the lip, the chin) stay on the midline and do not print as horizontal
        # bands across the cheeks (B2a: concave bands at -45 and -25 deg elevation, 45 deg round)
        self.rf_s, self.sig_a = None, None
        if sp.get("front_smooth"):
            sb, sa = sp["front_smooth"]
            self.rf_s = self._smoothed(self.rf, sb)
            self.sig_a = math.radians(sa)
        if sp.get("side_smooth"):
            # B2d: the front view's side radius smoothed along the elevation too (degrees): the traced cheek-to-chin outline is
            # nearly straight, and its spline's small wiggles were CONCAVE along the elevation (-3..-5 /m at -21 and -38 deg),
            # which printed as grooves round the cheek from below the eye to the jaw's corner (review 11 item 5); at 6 deg the
            # outline moves 0.44 mm at most and nowhere is it concave
            self.rs = self._smoothed(self.rs, sp["side_smooth"])

    @staticmethod
    def _smoothed(cv, sb):
        """A copy of the polar curve `cv` smoothed by a Gaussian of `sb` degrees along the elevation."""
        bb = np.radians(np.arange(-90.0, 90.01, 0.5))
        r = cv(bb)
        k = np.arange(-int(4 * sb / 0.5), int(4 * sb / 0.5) + 1)
        g = np.exp(-0.5 * (k * 0.5 / sb) ** 2)
        g /= g.sum()
        rp = np.concatenate([r[len(k) // 2:0:-1], r, r[-2:-len(k) // 2 - 2:-1]])
        return Curve(bb, np.convolve(rp, g, mode="same")[len(k) // 2:len(k) // 2 + len(bb)])

    def __call__(self, P):
        d = P - self.c
        R = np.linalg.norm(d, axis=1)
        D = d / np.maximum(R, 1e-9)[:, None]
        return R - self.rho(D)

    def rho(self, D):
        """The radius along unit directions D (n, 3)."""
        beta = np.arcsin(np.clip(D[:, 2], -1.0, 1.0))
        hx, hy = np.abs(D[:, 0]), D[:, 1]
        hh = np.maximum(np.hypot(hx, hy), 1e-9)
        sa, ca = hx / hh, -hy / hh
        front = ca >= 0.0
        rf = self.rf(beta)
        if self.rf_s is not None:
            w = np.exp(-(np.arctan2(sa, np.maximum(ca, 1e-9)) / self.sig_a) ** 2)
            rf = self.rf_s(beta) + (rf - self.rf_s(beta)) * w
        rfb = np.where(front, rf, self.rb(beta))
        n = np.where(front, self.nf(beta), self.nb(beta))
        return ((sa / self.rs(beta)) ** n + (np.abs(ca) / rfb) ** n) ** (-1.0 / n)


class Rings:
    """A tube of superellipse sections (z, centre y, half-width, front, back, exponent), natural cubic in z, capped
    at its first and last heights (C2)."""

    def __init__(self, rows, cap_k=0.004):
        cols = list(zip(*rows))
        z = np.asarray(cols[0], float)
        self.z0, self.z1 = z[0], z[-1]
        self.f = [Curve(z, c) for c in cols[1:]]
        self.k = cap_k

    def __call__(self, P):
        z = np.clip(P[:, 2], self.z0, self.z1)
        cy, a, bf, bb, n = (f(z) for f in self.f)
        dx, dy = np.abs(P[:, 0]), P[:, 1] - cy
        b = np.where(dy < 0.0, bf, bb)
        rho = ((dx / a) ** n + (np.abs(dy) / b) ** n) ** (1.0 / n)
        d = (rho - 1.0) * np.minimum(a, b)
        d = smax3(d, self.z0 - P[:, 2], self.k)
        return smax3(d, P[:, 2] - self.z1, self.k)


class Head:
    """The skin field and everything placed on it, from the `FACE` and `EYES` tables."""

    def __init__(self, face, eyes, scale=1.0):
        """`scale`: the whole head (everything but the neck) scaled about `face["pivot"]` (the chin's height on the
        neck's axis), the neck left as it is inside the collar: a head-size variant from the same tables. The tables'
        coordinates are HEAD SPACE; `__call__` / `base` are the world fields, `fh` / `base_h` the same in head space,
        `world()` maps head-space points out."""
        self.F, self.E = face, eyes
        self.egg = Egg(face["egg"])
        self.neck = Rings(face["neck"])
        eu, el = eyes["upper"], eyes["lower"]
        self.U = Curve([p[0] for p in eu], [p[1] for p in eu])
        self.L = Curve([p[0] for p in el], [p[1] for p in el])
        self.s = float(scale)
        self.o = np.asarray(face.get("pivot", (0.0, 0.005, 1.476)), float)

    def world(self, Q):
        return self.o + self.s * (np.asarray(Q, float) - self.o)

    def to_head(self, P):
        return self.o + (np.asarray(P, float) - self.o) / self.s

    def fh(self, Q):
        return self(self.world(Q)) / self.s

    def base_h(self, Q):
        return self.base(self.world(Q)) / self.s

    # --- pieces
    def bumps(self, P):
        Q = mirror_x(P)
        d = np.zeros(len(P))
        for c, s, h in self.F.get("bumps", ()):
            d += h * np.exp(-np.sum(((Q - np.asarray(c, float)) / np.asarray(s, float)) ** 2, axis=1))
        return d

    def nose(self, P):
        n = self.F["nose"]
        Q = mirror_x(P)
        d = capsule(Q, *n["bridge"])
        d = smin3(d, ellipsoid(Q, *n["tip"]), n["k_tip"])
        d = smin3(d, capsule(Q, *n["column"]), n["k_col"])
        d = smin3(d, ellipsoid(Q, *n["alae"]), n["k_alae"])
        return d, n["k"]

    def lips(self, P):
        m = self.F["mouth"]
        d = smin3(ellipsoid(P, *m["upper"]), ellipsoid(P, *m["lower"]), m["k_between"])
        return d, m["k"]

    def almond(self, P):
        """The eye opening in the front view (x, z): negative inside the drawn lid curves."""
        x, z = np.abs(P[:, 0]), P[:, 2]
        du = (z - self.U(x)) / np.sqrt(1.0 + self.U.d(x) ** 2)
        dl = (self.L(x) - z) / np.sqrt(1.0 + self.L.d(x) ** 2)
        return smax3(du, dl, self.E["corner_k"])

    def _forms(self, Q, nose=True, lips=True):
        """Head space, no neck, no cuts: egg + forms."""
        d = self.egg(Q) - self.bumps(Q)
        if nose:
            nd, nk = self.nose(Q)
            d = smin3(d, nd, nk)
        if lips:
            ld, lk = self.lips(Q)
            d = smin3(d, ld, lk)
        return d

    def shade_proxy(self, P):
        """World: the skin without the nose and with the lips flattened -- the surface whose normals the nose's bridge
        and sides and the lips are SHADED with (`shading` in the face table), as anime-style 3D faces are: the profile
        keeps the nose, the front view sees only its tip, as drawn."""
        P = np.asarray(P, float)
        Q = self.to_head(P)
        d = self._cut(Q, self._forms(Q, nose=False, lips=False)) * self.s
        return smin3(d, self.neck(P), self.F["neck_k"])

    def shading_normals(self, V):
        """Per vertex (world): the field's normals, blended towards the proxy's over the nose (all but its tip) and the
        lips by the face table's `shading` weights."""
        N = unit(self.grad(self, V))
        sh = self.F.get("shading")
        if not sh:
            return N
        Np = unit(self.grad(self.shade_proxy, V))
        Q = self.to_head(V)
        x, y, z = np.abs(Q[:, 0]), Q[:, 1], Q[:, 2]
        (z0, z1), sx, tip_c, tip_r, k_nose = sh["nose"]
        w = _ramp(z, z0, z0 + 0.003) * _ramp(z1 - z, 0.0, 0.003) * np.exp(-(x / sx) ** 2) * _ramp(-y, 0.080, 0.086)
        tip = np.exp(-((Q[:, 0] - tip_c[0]) / tip_r[0]) ** 2 - ((z - tip_c[1]) / tip_r[1]) ** 2)
        w = k_nose * w * (1.0 - tip)
        (lz0, lz1), lx, k_lip = sh["lips"]
        w = np.maximum(w, k_lip * _ramp(z, lz0, lz0 + 0.002) * _ramp(lz1 - z, 0.0, 0.002) * _ramp(lx - x, 0.0, 0.004) * _ramp(-y, 0.080, 0.084))
        return unit(N * (1.0 - w)[:, None] + Np * w[:, None])

    def toon_normals(self, V, sp):
        """B2b, for the toon look (`charkit/toonlook.py`): per vertex (world) the skin's field normals blended towards the
        EGG's -- the head's simple shape without the forms, the nose, the lips or the cuts -- as anime-style 3D faces are
        shaded, so the light falls on the face as on a smooth head (no nose, socket or cheek shadows; the profile keeps the
        nose's outline). `sp`: `weight` over the face, `front` (y0, y1) head space (full in front of y0, none behind y1),
        `z` (z0, z1, z2, z3) (none below z0, full z1-z2, none above z3), `tip` = ((x, z), (rx, rz), keep) the nose tip keeping
        `keep` of its own normal; `lift` = (amount, z0, z1): over the lower face (full below z0, none above z1) the normal is
        tipped up by `amount` (renormalised), so the jaw's side reads lit down to its edge and only its underside turns
        dark, as the art draws the jaw (a clean edge over the shadowed neck); `bias` = ((x, y, z), amount): over the face the
        normal is tipped towards that direction (head space) by `amount` and renormalised -- the anime face lit nearly
        whole by a key from the front, its shadow only at the far edge."""
        N = unit(self.grad(self, V))
        Ne = unit(self.grad(lambda P: self.egg(self.to_head(P)), V))
        Q = self.to_head(V)
        y0, y1 = sp["front"]
        z0, z1, z2, z3 = sp["z"]
        w = sp["weight"] * _ramp(y1 - Q[:, 1], 0.0, y1 - y0) * _ramp(Q[:, 2], z0, z1) * _ramp(z3 - Q[:, 2], 0.0, z3 - z2)
        if sp.get("tip"):
            (tx, tz), (rx, rz), keep = sp["tip"]
            tip = np.exp(-((Q[:, 0] - tx) / rx) ** 2 - ((Q[:, 2] - tz) / rz) ** 2) * _ramp(-Q[:, 1], 0.090, 0.097)
            w = w * (1.0 - keep * tip)
        N = unit(N * (1.0 - w)[:, None] + Ne * w[:, None])
        if sp.get("bias"):
            d, a = sp["bias"]
            N = unit(N + (a * w)[:, None] * unit(np.asarray(d, float))[None, :])
        if sp.get("lift"):
            a, lz0, lz1 = sp["lift"]
            k = a * _ramp(lz1 - Q[:, 2], 0.0, lz1 - lz0) * _ramp(Q[:, 2], sp["z"][0], sp["z"][1]) * _ramp(y1 - Q[:, 1], 0.0, y1 - y0)
            N = unit(N + np.column_stack([np.zeros(len(N)), np.zeros(len(N)), k]))
        return N

    def line_mask(self, V, sp):
        """B2b, for the toon look's inked outline (`toonlook.line_group`): per vertex (world) how much of the line the
        drawn face leaves out when seen from the front -- over the nose (`nose` = (amount, (z0, z1))) and the lips (`lips` =
        (amount, (z0, z1), half-width)); 0 elsewhere. The profile keeps every line."""
        Q = self.to_head(V)
        x, y, z = np.abs(Q[:, 0]), Q[:, 1], Q[:, 2]
        m = np.zeros(len(V))
        if sp.get("nose"):
            k, (z0, z1) = sp["nose"]
            m = np.maximum(m, k * _ramp(z, z0, z0 + 0.002) * _ramp(z1 - z, 0.0, 0.003) * _ramp(0.014 - x, 0.0, 0.004)
                           * _ramp(-y, 0.078, 0.084))
        if sp.get("lips"):
            k, (z0, z1), hw = sp["lips"]
            m = np.maximum(m, k * _ramp(z, z0, z0 + 0.002) * _ramp(z1 - z, 0.0, 0.002) * _ramp(hw - x, 0.0, 0.004)
                           * _ramp(-y, 0.078, 0.083))
        return m

    def _cut(self, Q, d):
        m = self.F["mouth"]
        d = smax3(d, -ellipsoid(Q, *m["slot"]), m["slot_k"])
        cut = smax3(self.almond(Q), -(self.egg(Q) - self.bumps(Q)) - self.E["depth"], self.E["depth_k"])
        if self.E.get("front_only") is not None:
            # B2c: the opening's prism runs along the view axis through the whole head, so it ALSO cut an eye-shaped dent into
            # the back of the skull (hidden under the hair; review 11's "ghost ear" in the hairless profile): kept to the front
            cut = smax3(cut, Q[:, 1] - self.E["front_only"], 0.002)
        return smax3(d, -cut, self.E["lid_k"])

    def base(self, P):
        """World: the skin without the cuts (egg + forms, then the neck)."""
        P = np.asarray(P, float)
        return smin3(self._forms(self.to_head(P)) * self.s, self.neck(P), self.F["neck_k"])

    def __call__(self, P):
        """World: the skin."""
        P = np.asarray(P, float)
        Q = self.to_head(P)
        d = self._cut(Q, self._forms(Q)) * self.s
        return smin3(d, self.neck(P), self.F["neck_k"])

    # --- helpers
    def box(self):
        """The skin's meshing box (world): B2a's head-space box, grown with the head-size scale (B2b: at x 1.20 the fixed box
        cut the crown flat at 1.692 and clipped the nose's tip)."""
        lo, hi = np.array([-0.095, -0.125, 1.395]), np.array([0.095, 0.100, 1.692])
        return np.minimum(lo, self.world(lo)), np.maximum(hi, self.world(hi))

    def shadow_proxy(self, h=0.002, shrink=0.006):
        """B2b, for the toon look: the head's SIMPLE shape (the egg and the neck, no forms, nose, lips or cuts) shrunk by
        `shrink`, meshed coarsely -- a shadow caster standing in for the skin (`toonlook.shadow_proxy`), so the chin still
        shades the neck and the head the hair, but no nose, lid or socket shadow falls on the face. Returns (verts, quads)."""
        def f(P):
            P = np.asarray(P, float)
            return smin3(self.egg(self.to_head(P)) * self.s, self.neck(P), self.F["neck_k"]) + shrink
        lo, hi = self.box()
        return mesh_banded(f, lo, hi, h)

    def grad(self, f, V, e=2e-5):
        return np.stack([(f(V + e * u) - f(V - e * u)) / (2 * e) for u in np.eye(3)], axis=1)

    def hit_front(self, f, x, z, y0=-0.14, y1=0.0, it=40):
        """Per (x, z), the first y where f turns negative marching from the front (bisection inside [y0, y1])."""
        x, z = np.asarray(x, float).ravel(), np.asarray(z, float).ravel()
        lo, hi = np.full(len(x), y0), np.full(len(x), y1)
        for _ in range(it):
            m = 0.5 * (lo + hi)
            ins = f(np.column_stack([x, m, z])) < 0.0
            lo, hi = np.where(ins, lo, m), np.where(ins, m, hi)
        return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------ meshing
def mesh_banded(field, lo, hi, h, block=4, margin=2.5, chunk=400000, project_iters=5):
    """The zero surface of `field` inside lo..hi at pitch h, sampled finely only in blocks of `block` cells whose
    coarse corners lie within `margin` coarse pitches of the surface. Surface nets on the crossed edges only (memory
    proportional to the surface), then Newton projection onto the field. Returns (verts, quads), outward-facing,
    closed when the field is positive on the box's border."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    nb = np.ceil((hi - lo) / (h * block)).astype(int)
    n = nb * block + 1
    H = h * block
    # coarse samples at the block corners
    gc = [lo[i] + H * np.arange(nb[i] + 1) for i in range(3)]
    Pc = np.stack(np.meshgrid(*gc, indexing="ij"), axis=-1).reshape(-1, 3)
    Fc = np.concatenate([field(Pc[i:i + chunk]) for i in range(0, len(Pc), chunk)]).reshape(nb + 1)
    near = np.abs(Fc) < margin * H
    act = np.zeros(nb, bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                act |= near[dx:dx + nb[0], dy:dy + nb[1], dz:dz + nb[2]]
    # the fine grid: the coarse value of the block's nearest corner everywhere, exact in the active blocks
    F = np.empty(tuple(n), np.float32)
    ci = [np.minimum(np.round(np.arange(n[i]) / block).astype(int), nb[i]) for i in range(3)]
    F[...] = Fc[np.ix_(ci[0], ci[1], ci[2])]
    blocks = np.argwhere(act)
    o = np.arange(block + 1)
    BI, BJ, BK = np.meshgrid(o, o, o, indexing="ij")
    off = np.stack([BI.ravel(), BJ.ravel(), BK.ravel()], axis=1)
    per = len(off)
    step = max(1, chunk // per)
    for s in range(0, len(blocks), step):
        idx = (blocks[s:s + step, None, :] * block + off[None, :, :]).reshape(-1, 3)
        F[idx[:, 0], idx[:, 1], idx[:, 2]] = field(lo + h * idx).astype(np.float32)
    F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (np.abs(F[0]) + h, np.abs(F[-1]) + h, np.abs(F[:, 0]) + h,
                                                                np.abs(F[:, -1]) + h, np.abs(F[:, :, 0]) + h, np.abs(F[:, :, -1]) + h)
    V, Q = _nets_sparse(F, lo, h)
    V = _project(field, V, h, project_iters)
    return V, Q


def _nets_sparse(F, lo, h):
    ins = F < 0.0
    nx, ny, nz = F.shape
    cs = np.array([(ny - 1) * (nz - 1), nz - 1, 1], np.int64)       # cell linear index strides
    cell_ids, cell_pts, quads_raw = [], [], []
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax], s1[ax] = slice(0, -1), slice(1, None)
        e = ins[tuple(s0)] != ins[tuple(s1)]
        idx = np.argwhere(e)                                        # edge (i, j, k) from point idx to idx + e_ax
        f0 = F[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
        i1 = idx.copy()
        i1[:, ax] += 1
        f1 = F[i1[:, 0], i1[:, 1], i1[:, 2]].astype(np.float64)
        t = f0 / (f0 - f1)
        p = idx.astype(np.float64)
        p[:, ax] += t
        o1, o2 = [a for a in range(3) if a != ax]
        if (o1, o2) == (0, 2):
            o1, o2 = 2, 0
        dims = np.array([nx - 1, ny - 1, nz - 1])
        cells4 = []
        for d1, d2 in ((1, 1), (0, 1), (0, 0), (1, 0)):
            c = idx.copy()
            c[:, o1] -= d1
            c[:, o2] -= d2
            cells4.append(c)
        ok = np.ones(len(idx), bool)
        for c in cells4:
            ok &= np.all((c >= 0) & (c < dims), axis=1)
        for c in cells4:
            lid = c @ cs
            cell_ids.append(lid[ok])
            cell_pts.append(p[ok])
        q = np.stack([c @ cs for c in cells4], axis=1)[ok]
        start_in = ins[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        quads_raw.append(np.where(start_in[:, None], q, q[:, ::-1]))
    ids = np.concatenate(cell_ids)
    pts = np.concatenate(cell_pts)
    uniq, inv = np.unique(ids, return_inverse=True)
    acc = np.zeros((len(uniq), 3))
    np.add.at(acc, inv, pts)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    V = acc / cnt[:, None]
    Qr = np.vstack(quads_raw)
    Q = np.searchsorted(uniq, Qr)
    return lo + h * V, Q


def _project(field, V, h, iters):
    V = np.array(V, float)
    e = 0.25 * h
    for _ in range(iters):
        f = field(V)
        g = np.stack([(field(V + e * u) - field(V - e * u)) / (2 * e) for u in np.eye(3)], axis=1)
        gg = np.maximum(np.sum(g * g, axis=1), 1e-12)
        step = (f / gg)[:, None] * g
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        V -= step * np.minimum(1.0, h / np.maximum(nrm, 1e-12))
    return V


# ------------------------------------------------------------------------------------------------ sweeps
def sweep(pts, side, ra, rb, n_u=10, cap=True):
    """A closed tube along points (m, 3) with an elliptical section: half-axis `ra` (m,) along `side` (m, 3, made
    perpendicular to the path) and `rb` (m,) along the third axis. The ends close in a point (a cap vertex).
    Returns (verts, quads, tris)."""
    pts = np.asarray(pts, float)
    m = len(pts)
    T = unit(np.gradient(pts, axis=0))
    S = unit(side - np.sum(side * T, axis=1)[:, None] * T)
    W = np.cross(T, S)
    al = np.linspace(0.0, 2 * math.pi, n_u, endpoint=False)
    ra, rb = np.broadcast_to(ra, (m,)), np.broadcast_to(rb, (m,))
    V = pts[:, None, :] + (ra[:, None, None] * np.cos(al)[None, :, None] * S[:, None, :]
                           + rb[:, None, None] * np.sin(al)[None, :, None] * W[:, None, :])
    V = V.reshape(-1, 3)
    idx = np.arange(m * n_u).reshape(m, n_u)
    nx = np.roll(idx, -1, axis=1)
    Q = np.stack([idx[:-1], nx[:-1], nx[1:], idx[1:]], axis=-1).reshape(-1, 4)
    tris = np.zeros((0, 3), int)
    if cap:
        a, b = len(V), len(V) + 1
        V = np.vstack([V, pts[0] - 1e-5 * T[0], pts[-1] + 1e-5 * T[-1]])
        tris = np.vstack([np.stack([np.full(n_u, a), nx[0], idx[0]], axis=-1), np.stack([np.full(n_u, b), idx[-1], nx[-1]], axis=-1)])
    # outward: check one face against the section's radial direction
    f = Q[len(Q) // 2]
    nrm = np.cross(V[f[1]] - V[f[0]], V[f[3]] - V[f[0]])
    i_row = len(Q) // 2 // n_u
    if nrm @ (V[f[0]] - pts[i_row]) < 0:
        Q, tris = Q[:, ::-1], tris[:, ::-1]
    return V, Q, tris


# ------------------------------------------------------------------------------------------------ masks
def _band(v, c, w):
    return np.exp(-((v - c) / w) ** 2)


def _ramp(v, a, b):
    q = np.clip((v - a) / (b - a), 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q)


def face_masks(head, V):
    """The skin's painted masks, per vertex, from the front-projected (|x|, z) and only on the face's front:
    blush, lips, the mouth line, the lower lash line, the lid crease, brows, nostrils, the nose tip."""
    F = head.F
    pm = F["paint"]
    x, y, z = np.abs(V[:, 0]), V[:, 1], V[:, 2]
    front = _ramp(-y, 0.040, 0.060)
    out = {}
    bc, bs = pm["blush"]
    out["blush"] = np.exp(-((x - bc[0]) / bs[0]) ** 2 - ((z - bc[1]) / bs[1]) ** 2) * front
    m = F["mouth"]
    lip = np.zeros(len(V))
    for c, r in (m["upper"], m["lower"]):
        q = ((V[:, 0] - c[0]) / (r[0] * 1.05)) ** 2 + ((z - c[2]) / (r[2] * 1.15)) ** 2
        lip = np.maximum(lip, _ramp(1.0 - q, 0.0, 0.5))
    out["lips"] = lip * _ramp(-y, 0.080, 0.084)
    mz, mw, mh = pm["mouth_line"]
    out["mouthline"] = _band(z, mz, mh) * _ramp(mw - x, 0.0, 0.004) * front
    x0, x1, off, w = pm["low_lash"]
    zl = head.L(x) - off
    span = _ramp(x, x0, x0 + 0.006) * _ramp(x1 - x, 0.0, 0.003)
    spikes = 0.85 + 0.15 * np.cos(x * 2 * math.pi / 0.0024)
    out["lowlash"] = _band(z, zl, w) * span * front * spikes
    x0, x1, off, w = pm["crease"]
    out["crease"] = _band(z, head.U(x) + off, w) * _ramp(x, x0, x0 + 0.006) * _ramp(x1 - x, 0.0, 0.006) * front
    bx0, bx1, bz, arch, bw = pm["brow"]
    u = np.clip((x - bx0) / (bx1 - bx0), 0.0, 1.0)
    zb = bz + arch * np.sin(math.pi * u ** 0.8)
    out["brow"] = _band(z, zb, bw * (1.0 - 0.6 * u)) * _ramp(x, bx0, bx0 + 0.004) * _ramp(bx1 - x, 0.0, 0.008) * front
    nx, nz, nr = pm["nostril"]
    out["nostril"] = np.exp(-((x - nx) / nr[0]) ** 2 - ((z - nz) / nr[1]) ** 2) * _ramp(-y, 0.085, 0.090)
    tz, tr = pm["nose_tip"]
    out["nosetip"] = np.exp(-(V[:, 0] / tr[0]) ** 2 - ((z - tz) / tr[1]) ** 2) * _ramp(-y, 0.095, 0.100)
    if pm.get("blush_bridge"):   # B2d: the blush across the nose's bridge, as drawn (one band under the eyes)
        (bz, (sx, sz)), k = pm["blush_bridge"]
        out["blush"] = np.maximum(out["blush"], k * np.exp(-(x / sx) ** 2 - ((z - bz) / sz) ** 2) * front)
    art = F.get("art")
    if art:   # B2d: where the drawn layer (`face_art`) lies on the skin: its image coordinates, its weight
        x1, z0, z1 = art["frame"]
        out["art_u"] = x / x1
        out["art_v"] = (z - z0) / (z1 - z0)
        out["art_w"] = _ramp(-y, *art.get("front", (0.040, 0.055)))
    return out


# ------------------------------------------------------------------------------------------------ the drawn layer (B2d)
def _srgb(c):
    h = c.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


class Canvas:
    """A front-projected paint layer in head space (B2d): columns |x| from 0 to `x1`, rows z from `z1` down to `z0` at
    pitch `px` (m); straight-alpha RGBA, composited layer by layer ('over'), every edge anti-aliased by its distance."""

    def __init__(self, x1, z0, z1, px):
        self.px, self.x1, self.z0, self.z1 = px, x1, z0, z1
        self.W, self.H = int(math.ceil(x1 / px)), int(math.ceil((z1 - z0) / px))
        self.rgb = np.zeros((self.H, self.W, 3), np.float32)
        self.a = np.zeros((self.H, self.W), np.float32)

    def window(self, lo, hi):
        j0, j1 = max(0, int(lo[0] / self.px) - 1), min(self.W, int(hi[0] / self.px) + 2)
        i0, i1 = max(0, int((self.z1 - hi[1]) / self.px) - 1), min(self.H, int((self.z1 - lo[1]) / self.px) + 2)
        xs = (np.arange(j0, j1) + 0.5) * self.px
        zs = self.z1 - (np.arange(i0, i1) + 0.5) * self.px
        return (i0, i1, j0, j1), xs, zs

    def over(self, win, cov, colour, alpha=1.0):
        i0, i1, j0, j1 = win
        a = np.clip(cov * alpha, 0.0, 1.0).astype(np.float32)
        if a.size == 0:
            return
        A, C = self.a[i0:i1, j0:j1], self.rgb[i0:i1, j0:j1]
        na = a + A * (1.0 - a)
        C[...] = (np.asarray(colour, np.float32)[None, None, :] * a[..., None] + C * (A * (1.0 - a))[..., None]) \
            / np.maximum(na, 1e-6)[..., None]
        A[...] = na

    def field(self, fn, lo, hi, colour, alpha=1.0):
        """A soft shape: `fn`(X, Z) -> coverage 0..1 over the box lo..hi."""
        win, xs, zs = self.window(lo, hi)
        if len(xs) and len(zs):
            X, Z = np.meshgrid(xs, zs)
            self.over(win, fn(X, Z), colour, alpha)

    def stroke(self, pts, w, colour, alpha=1.0, soft=1.0):
        """A polyline (n, 2) of widths `w` (n,) (a taper), its edge `soft` pixels wide."""
        pts = np.asarray(pts, float)
        w = np.broadcast_to(np.asarray(w, float), (len(pts),))
        pad = w.max() / 2 + (2 + soft) * self.px
        win, xs, zs = self.window(pts.min(0) - pad, pts.max(0) + pad)
        if not (len(xs) and len(zs)):
            return
        X, Z = np.meshgrid(xs, zs)
        P = np.stack([X.ravel(), Z.ravel()], 1)
        best = np.full(len(P), -1e9)
        for k in range(len(pts) - 1):
            a, ab = pts[k], pts[k + 1] - pts[k]
            t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-18), 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
            best = np.maximum(best, ((w[k] + (w[k + 1] - w[k]) * t) / 2 - d) / (self.px * soft) + 0.5)
        self.over(win, np.clip(best, 0.0, 1.0).reshape(X.shape), colour, alpha)

    def image(self, name):
        """The layer as a packed Blender image (sRGB, straight alpha)."""
        import bpy
        img = bpy.data.images.new(name, self.W, self.H, alpha=True)
        img.alpha_mode = "STRAIGHT"
        px = np.concatenate([self.rgb, self.a[..., None]], axis=2)[::-1]      # Blender's rows run bottom to top
        img.pixels.foreach_set(np.ascontiguousarray(px, np.float32).ravel())
        img.pack()
        return img


def _lash(start, ang, length, bend, n=8):
    """A drawn lash: from `start` at `ang` deg from vertical (+ outwards), curving by `bend` deg along it."""
    pts = [np.asarray(start, float)]
    for k in range(1, n + 1):
        a = math.radians(ang + bend * (k / n) ** 2)
        pts.append(pts[-1] + (length / n) * np.array([math.sin(a), math.cos(a)]))
    return np.array(pts)


def face_art(head, art, p):
    """B2d: the face's DRAWN layer -- what the close-ups ink and paint rather than model -- as one front-projected canvas
    (head space, |x|: both sides alike), every shape from the `art` table and every colour from the palette `p`:
    the upper lash MASS (the band's core, a soft brown fill and feathered lash strokes along its top, the clumped lashes
    fanning out at the outer end), the double-lid crease, the lower lid's light rim, its lash line and lower lashes, the
    inner corner's pink, the brows (a soft body and hair strokes), the nose's highlight (a soft teardrop and a glint) and
    nostril strokes, the mouth line, the lower lip's tint and gloss. Returns the Canvas."""
    rng = np.random.default_rng(art.get("seed", 1))
    x1, z0, z1 = art["frame"]
    cv = Canvas(x1, z0, z1, art["px"])
    c = lambda k: _srgb(p[k])
    U, Lw = head.U, head.L
    # --- the inner corner and the lower lid (under the lashes)
    if art.get("inner"):
        (ix, iz), (sx, sz), a = art["inner"]
        cv.field(lambda X, Z: np.exp(-((X - ix) / sx) ** 2 - ((Z - iz) / sz) ** 2), (ix - 3 * sx, iz - 3 * sz),
                 (ix + 3 * sx, iz + 3 * sz), c("INNER_CORNER"), a)
    lo = art.get("lower")
    if lo:
        xs = np.linspace(lo["x"][0], lo["x"][1], 50)
        uu = (xs - xs[0]) / (xs[-1] - xs[0])
        r0, r1, ra = lo["rim"]          # the lid's light rim: from r0 to r1 under the lid line
        xr = np.linspace(lo["x"][0], lo.get("rim_x", lo["x"][1]), 50)
        ur = (xr - xr[0]) / (xr[-1] - xr[0])
        cv.stroke(np.column_stack([xr, Lw(xr) - 0.5 * (r0 + r1)]), (r1 - r0) * np.sin(np.pi * np.clip(ur, 0.02, 0.98)) ** 0.3,
                  c("LID_RIM"), ra, soft=2.0)
        off, lw, la = lo["line"]       # the lower lash line, fading in from the inner end
        cv.stroke(np.column_stack([xs, Lw(xs) - off]), lw * np.interp(uu, *zip(*lo["taper"])), c("LASH_LOW"), la, soft=1.5)
        for xr, ln, ang, wr in lo.get("lashes", ()):
            cv.stroke(_lash((xr, float(Lw(xr)) - off), 180.0 - ang, ln, -lo.get("bend", 8.0), 6),
                      np.linspace(wr, wr * 0.2, 7), c("LASH_LOW"), la, soft=1.2)
    if art.get("outer"):   # the outer corner's walls inked dark: the white tucks under the lids there, as drawn
        pts, wd, oa = art["outer"]
        cv.stroke(np.asarray(pts, float), np.interp(np.linspace(0, 1, len(pts)), *zip(*wd)), c("LASH"), oa, soft=1.5)
    # --- the crease above the lid
    cr = art.get("crease")
    if cr:
        xs = np.linspace(cr["x"][0], cr["x"][1], 50)
        uu = (xs - xs[0]) / (xs[-1] - xs[0])
        cv.stroke(np.column_stack([xs, U(xs) + np.interp(uu, *zip(*cr["off"]))]), cr["width"] * np.sin(np.pi * uu) ** 0.5,
                  c("CREASE"), cr.get("alpha", 1.0), soft=1.5)
    # --- the upper lash mass
    la = art.get("lash")
    if la:
        X, Zc, w, u = lash_path(head)
        grow = la.get("grow", 1.0)
        pad = la.get("pad", 0.0)
        core = w * grow + (np.interp(u, *zip(*pad)) if isinstance(pad, (tuple, list)) else pad)
        top = np.column_stack([X, Zc + core / 2])
        f_w, f_a = la["fill"]          # the soft brown above the core (the strokes' roots)
        cv.stroke(np.column_stack([X, Zc + core / 2 + f_w * 0.25]), np.full(len(X), f_w) * np.sin(np.pi * np.clip(u, 0.03, 0.97)) ** 0.5,
                  c("LASH_SOFT"), f_a, soft=f_w / art["px"] * 0.35)
        cv.stroke(np.column_stack([X, Zc]), core, c("LASH"), 1.0, soft=1.2)
        st = la["strokes"]
        u_s = np.arange(st["u"][0], st["u"][1], st["pitch"])
        u_s = np.clip(u_s + rng.uniform(-0.4, 0.4, len(u_s)) * st["pitch"], 0.0, 1.0)
        for us in u_s:
            root = np.array([np.interp(us, u, top[:, 0]), np.interp(us, u, top[:, 1]) - 0.35 * np.interp(us, u, core)])
            ln = np.interp(us, *zip(*st["length"])) * rng.uniform(0.65, 1.25)
            ang = np.interp(us, *zip(*st["angle"])) + rng.uniform(-8.0, 8.0)
            wd = st["width"][0] * rng.uniform(0.8, 1.2)
            cv.stroke(_lash(root, ang, ln + 0.35 * np.interp(us, u, core), st.get("bend", 10.0)),
                      np.linspace(wd, st["width"][1], 9), c("LASH"), 1.0, soft=1.2)
        for us, ln, ang, bend, wd in la.get("clumps", ()):   # the outer lashes, drawn one by one
            root = np.array([np.interp(us, u, X), np.interp(us, u, Zc)])
            cv.stroke(_lash(root, ang, ln, bend, 12), np.linspace(wd, wd * 0.15, 13), c("LASH"), 1.0, soft=1.2)
    # --- the brows: a soft body and hair strokes along it
    br = art.get("brow")
    if br:
        pts = np.asarray(br["pts"], float)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        ss = np.linspace(0.0, s[-1], 60)
        line = np.column_stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])])
        uu = ss / s[-1]
        wb = np.interp(uu, *zip(*br["width"]))
        cv.stroke(line, wb, c("BROW"), br.get("body", 0.85), soft=2.5)
        hs = br["hairs"]
        n = int(hs["n"])
        for k in range(n):
            q = rng.uniform(0.0, 0.97)
            i = int(q * (len(line) - 1))
            d = line[min(i + 1, len(line) - 1)] - line[max(i - 1, 0)]
            ang = math.degrees(math.atan2(d[0], d[1])) - hs.get("lift", 12.0) * (1.0 - q) + rng.uniform(-6.0, 6.0)
            across = rng.uniform(-0.5, 0.5) * np.interp(q, uu, wb)
            nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            root = line[i] + across * nrm - 0.5 * hs["length"] * d / max(np.linalg.norm(d), 1e-9)
            cv.stroke(_lash(root, ang, hs["length"] * rng.uniform(0.6, 1.1), 0.0, 4), np.linspace(hs["width"], hs["width"] * 0.3, 5),
                      c("BROW"), 0.9, soft=1.2)
    # --- the nose: a soft teardrop highlight (narrow at the top), its glint, the nostril strokes
    no = art.get("nose")
    if no:
        zt, zb, hw, edge, a = no["hi"]

        def tear(X, Z):
            t = np.clip((zt - Z) / (zt - zb), 0.0, 1.0)
            half = hw * np.sin(np.pi * np.clip(t, 0.0, 1.0) ** 0.42) ** 0.55
            inside = np.clip((half - X) / edge + 0.5, 0.0, 1.0)
            return inside * np.clip((zt - Z) / edge + 0.5, 0, 1) * np.clip((Z - zb) / edge + 0.5, 0, 1)
        cv.field(tear, (0.0, zb - edge), (hw + edge, zt + edge), c("NOSE_HI"), a)
        (gx, gz), (gsx, gsz), ga = no["glint"]
        cv.field(lambda X, Z: np.clip(1.5 - np.sqrt(((X - gx) / gsx) ** 2 + ((Z - gz) / gsz) ** 2) * 1.5, 0, 1),
                 (0.0, gz - gsz * 2), (gx + gsx * 2, gz + gsz * 2), c("NOSE_GLINT"), ga)
        pts, wd, na = no["nostril"]
        pts = np.asarray(pts, float)
        cv.stroke(pts, np.interp(np.linspace(0, 1, len(pts)), *zip(*wd)), c("NOSTRIL" if "NOSTRIL" in p else "LIP_LINE"), na, soft=1.5)
    # --- the mouth: the lower lip's tint and gloss, the line
    mo = art.get("mouth")
    if mo:
        (tz, (tsx, tsz)), ta = mo["tint"]
        cv.field(lambda X, Z: np.exp(-(X / tsx) ** 4 - ((Z - tz) / tsz) ** 2), (0.0, tz - 3 * tsz), (tsx * 1.6, tz + 3 * tsz), c("LIP"), ta)
        if mo.get("upper"):   # the upper lip's faint tint over the line
            (uz, (usx, usz)), ua = mo["upper"]
            cv.field(lambda X, Z: np.exp(-(X / usx) ** 4 - ((Z - uz) / usz) ** 2), (0.0, uz - 3 * usz), (usx * 1.6, uz + 3 * usz), c("LIP"), ua)
        for (gx, gz), (gsx, gsz), ga in mo.get("gloss", ()):
            cv.field(lambda X, Z: np.exp(-((X - gx) / gsx) ** 2 - ((Z - gz) / gsz) ** 2), (0.0, gz - 3 * gsz),
                     (gx + 3 * gsx, gz + 3 * gsz), c("LIP_GLOSS"), ga)
        pts = np.asarray(mo["line"], float)
        cv.stroke(pts, np.interp(np.linspace(0, 1, len(pts)), *zip(*mo["width"])), c("MOUTH_LINE" if "MOUTH_LINE" in p else "LIP_LINE"),
                  mo.get("alpha", 0.9), soft=mo.get("soft", 2.0))
    return cv


# ------------------------------------------------------------------------------------------------ materials (bpy)
def materials(prefix, p, eyes=None, face=None):
    """The head's own materials, every colour from the palette `p`: `<prefix>_face` (the skin with its painted masks;
    the same base, subsurface and gloss as the hands' `<prefix>_skin`, no pore bump: the drawn skin is clean),
    `_eyes` (sclera, the drawn iris and its highlights from the eye surface's `iu`/`iv`/`lid`), `_lash`, `_hair`.
    B2d: with `face` carrying an `art` table, the skin wears the drawn layer (`face_art`, an image over `art_u`/`art_v`)
    in place of B2a/B2b's painted brow, crease, lower lash line, nostrils, mouth line and nose highlight."""
    from studio import NT
    from wearmat import lin
    import bpy

    def new(name):
        m = bpy.data.materials.new(prefix + "_" + name)
        return m, NT(m)

    def mixc(t, fac, a, b):
        return t.mix(fac, a, b)

    out = {}
    # --- skin
    m, t = new("face")
    s = lin(p["SKIN"])
    col = t.mix(t.math("MULTIPLY", t.ramp(t.noise(14.0, 2.0, 0.5), 0.35, 0.75), 0.25), s, (s[0] * 1.03, s[1] * 0.93, s[2] * 0.90, 1.0))
    col = t.mix(t.math("MULTIPLY", t.attr("blush"), p.get("BLUSH_AMOUNT", 0.55)), col, lin(p["BLUSH"]))
    col = t.mix(t.math("MULTIPLY", t.attr("nosetip"), p.get("NOSE_WARM", 0.35)), col, lin(p["BLUSH"]))
    art = (face or {}).get("art")
    if "NOSE_HI" in p and not art:   # B2b: the drawn tip highlight (a pale soft shape) over the tip's warmth
        col = t.mix(t.math("MULTIPLY", t.ramp(t.attr("nosetip"), 0.25, 0.8), 0.75), col, lin(p["NOSE_HI"]))
    col = t.mix(t.math("MULTIPLY", t.attr("lips"), p.get("LIP_AMOUNT", 0.45)), col, lin(p["LIP"]))
    if art:   # B2d: the drawn layer (`face_art`) over the skin
        img = face_art(Head(face, eyes), art, p).image(prefix + "_face_art")
        tex = t.n("ShaderNodeTexImage")
        tex.image, tex.interpolation, tex.extension = img, "Linear", "CLIP"
        uv = t.n("ShaderNodeCombineXYZ")
        t.put(uv.inputs[0], t.attr("art_u"))
        t.put(uv.inputs[1], t.attr("art_v"))
        t.put(tex.inputs["Vector"], uv.outputs[0])
        col = t.mix(t.math("MULTIPLY", tex.outputs["Alpha"], t.attr("art_w")), col, tex.outputs["Color"])
    else:
        col = t.mix(t.math("MULTIPLY", t.attr("crease"), p.get("CREASE_AMOUNT", 0.55)), col, lin(p["CREASE"]))
        col = t.mix(t.math("MULTIPLY", t.attr("lowlash"), 0.75), col, lin(p["LASH_LOW"]))
        strands = t.ramp(t.noise(300.0, 2.0, 0.5, 0.0, "FBM", stretch=(1.0, 1.0, 0.3)), 0.3, 0.65)
        # B2b: the brow's band sharpened (a ramp on its mask): B2a's Gaussian read as a blurred smudge in the toon look
        col = t.mix(t.math("MULTIPLY", t.ramp(t.attr("brow"), 0.25, 0.6), t.math("ADD", t.math("MULTIPLY", strands, 0.25), 0.72), clamp=True),
                    col, lin(p["BROW"]))
        col = t.mix(t.math("MULTIPLY", t.attr("nostril"), p.get("NOSTRIL_AMOUNT", 0.35)), col, lin(p["LIP_LINE"]))
        col = t.mix(t.math("MULTIPLY", t.attr("mouthline"), 0.85), col, lin(p["LIP_LINE"]))
    ao = t.ramp(t.ao(0.02), 0.2, 0.95)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, ao), 0.25), col, (s[0] * 0.55, s[1] * 0.40, s[2] * 0.38, 1.0))
    rough = t.mixf(t.attr("lips"), 0.45, 0.28)
    rough = t.mixf(t.attr("nosetip"), rough, 0.30)
    t.set(Base_Color=col, Roughness=rough, Subsurface_Weight=0.2, Subsurface_Radius=(1.0, 0.45, 0.28), Subsurface_Scale=0.004)
    m["game_gloss"] = 0.0
    out["face"] = m
    # --- eyes
    m, t = new("eyes")
    iu, iv = t.attr("iu"), t.attr("iv")

    def dist(a, b, sx=1.0, sy=1.0):
        dx = t.math("DIVIDE", t.math("SUBTRACT", iu, a), sx)
        dy = t.math("DIVIDE", t.math("SUBTRACT", iv, b), sy)
        return t.math("SQRT", t.math("ADD", t.math("MULTIPLY", dx, dx), t.math("MULTIPLY", dy, dy)))

    r = dist(0.0, 0.0)
    sc = lin(p["SCLERA"])
    ep = (eyes or {}).get("paint")   # B2d: the drawn eye's layers (None: B2b's)
    if ep:
        # the sclera: grey under the upper lid's shadow (a soft band from the lid down), greyer still where it tucks under
        # the lids at the corners (`ec`, the opening's distance in mm), the inner corner pink (`ex` 0 at the inner end)
        s0, s1 = ep["sclera_shadow"]
        col = t.mix(t.math("SUBTRACT", 1.0, t.ramp(t.attr("lid"), s0, s1)), sc, lin(p["SCLERA_SHADE"]))
        c0, c1, ca = ep["corner"]
        col = t.mix(t.math("MULTIPLY", t.ramp(t.attr("ec"), c0, c1), ca), col, lin(p["SCLERA_SHADE"]))
        i0, i1, ia = ep["inner"]
        col = t.mix(t.math("MULTIPLY", t.ramp(t.attr("ex"), i1, i0), ia), col, lin(p["INNER_CORNER"]))
    else:
        col = t.mix(t.math("SUBTRACT", 1.0, t.ramp(t.attr("lid"), 0.0, 0.55)), sc, lin(p["SCLERA_SHADE"]))
    ir_dark, ir_mid, ir_glow = lin(p["IRIS_DARK"]), lin(p["IRIS"]), lin(p["IRIS_GLOW"])
    top = ep["top"] if ep else (-0.05, 0.55)
    irc = t.mix(t.ramp(iv, *top), ir_mid, ir_dark)
    low = ep["low"] if ep else (-0.15, -0.80)
    irc = t.mix(t.ramp(iv, *low), irc, ir_glow)
    cx = t.n("ShaderNodeCombineXYZ")
    t.put(cx.inputs[0], t.math("MULTIPLY", t.math("DIVIDE", iu, t.math("ADD", r, 0.02)), 6.0))
    t.put(cx.inputs[1], t.math("MULTIPLY", t.math("DIVIDE", iv, t.math("ADD", r, 0.02)), 6.0))
    t.put(cx.inputs[2], t.math("MULTIPLY", r, 1.5))
    streak = t.noise(3.0, 3.0, 0.6, 0.0, "FBM", vec=cx.outputs[0])
    irc = t.mix(t.math("MULTIPLY", t.ramp(streak, 0.35, 0.7), ep.get("streak", 0.35) if ep else 0.35), irc,
                (ir_glow[0], ir_glow[1], ir_glow[2], 1.0))
    rim = ep["rim"] if ep else (0.70, 0.97, 0.7)
    irc = t.mix(t.ramp(r, rim[0], rim[1]), irc, (ir_dark[0] * rim[2], ir_dark[1] * rim[2], ir_dark[2] * rim[2], 1.0))
    # B2b: the pupil's size from the eye table (`iris["pupil"]`, half-axes in iris radii; B2a's 0.30 x 0.36), a lighter ring
    # round it and a soft glow low on the far side, as the drawing layers its iris
    pr = (eyes or {}).get("iris", {}).get("pupil", (0.30, 0.36))
    pup = dist(0.0, 0.02, pr[0], pr[1])
    irc = t.mix(t.math("MULTIPLY", t.ramp(pup, 1.0, 1.25), t.ramp(pup, 2.2, 1.6)), irc,
                (ir_glow[0] * 0.8, ir_glow[1] * 0.8, ir_glow[2] * 0.85, 1.0))
    irc = t.mix(t.math("MULTIPLY", t.ramp(dist(0.25, -0.45, 0.55, 0.35), 1.0, 0.3), 0.55), irc, (ir_glow[0] * 1.25, ir_glow[1] * 1.25, ir_glow[2] * 1.2, 1.0))
    irc = t.mix(t.ramp(pup, 1.08, 0.88), irc, lin(p["PUPIL"]))
    col = t.mix(t.ramp(r, 1.03, 0.97), col, irc)
    # the lid's shadow over the iris's top as well
    sh = ep["iris_shadow"] if ep else (0.0, 0.30)
    col = t.mix(t.math("MULTIPLY", t.math("SUBTRACT", 1.0, t.ramp(t.attr("lid"), *sh)), p.get("LID_SHADOW", 0.55)), col, lin(p["IRIS_DARK"]))
    if ep:   # B2d: the highlights as drawn -- (u, v, radius, softness) in iris radii, the first the big one
        hl = None
        for hu, hv_, hr, hs in ep["highlights"]:
            h1 = t.ramp(dist(hu, hv_), hr + hs, hr)
            hl = h1 if hl is None else t.math("MAXIMUM", hl, h1)
    else:
        hl = t.math("MAXIMUM", t.ramp(dist(0.20, 0.40), 0.17, 0.13), t.ramp(dist(-0.07, 0.17), 0.075, 0.05))
    col = t.mix(hl, col, (1.0, 1.0, 1.0, 1.0))
    t.set(Base_Color=col, Roughness=0.25, Specular_IOR_Level=0.35, Coat_Weight=0.4, Coat_Roughness=0.08,
          Emission_Color=(1.0, 1.0, 1.0, 1.0), Emission_Strength=t.math("MULTIPLY", hl, 2.0))
    out["eyes"] = m
    # --- lashes
    m, t = new("lash")
    t.set(Base_Color=lin(p["LASH"]), Roughness=0.5, Specular_IOR_Level=0.3)
    out["lash"] = m
    # --- hair (a blockout's: dark brown, streaks across each lock, a soft sheen)
    m, t = new("hair")
    hc = lin(p["HAIR"])
    hv = t.attr("hv")
    cx = t.n("ShaderNodeCombineXYZ")
    t.put(cx.inputs[0], t.math("MULTIPLY", hv, 60.0))
    t.put(cx.inputs[1], t.math("MULTIPLY", t.attr("hu"), 1.5))
    streak = t.noise(4.0, 4.0, 0.6, 0.0, "FBM", vec=cx.outputs[0])
    hl = lin(p["HAIR_LIGHT"])
    col = t.mix(t.math("MULTIPLY", t.ramp(streak, 0.35, 0.75), p.get("HAIR_STREAK", 0.6)), hc, hl)
    if "HAIR_LINE" in p:   # B2e: the drawn strand lines -- thin dark lines along each lock (the isolines of a noise
        # stretched along it: `HAIR_LINE_SCALE` = (per metre across, along), `HAIR_LINE_W` their width in noise units)
        sx, sy = p.get("HAIR_LINE_SCALE", (400.0, 6.0))
        lx = t.n("ShaderNodeCombineXYZ")
        t.put(lx.inputs[0], t.math("MULTIPLY", hv, sx))
        t.put(lx.inputs[1], t.math("MULTIPLY", t.attr("hu"), sy))
        nz = t.noise(1.0, 2.0, 0.5, 0.0, "FBM", vec=lx.outputs[0])
        wl = p.get("HAIR_LINE_W", 0.025)
        line = t.ramp(t.math("ABSOLUTE", t.math("SUBTRACT", nz, 0.5)), wl, 0.0)
        col = t.mix(t.math("MULTIPLY", line, p.get("HAIR_LINE_AMOUNT", 0.7)), col, lin(p["HAIR_LINE"]))
    if "HAIR_DEEP" in p:   # B2c: the under layer and the cap darker (`hshade`, per lock), the depth under the outer locks
        col = t.mix(t.math("MULTIPLY", t.attr("hshade"), p.get("HAIR_DEEP_AMOUNT", 0.6)), col, lin(p["HAIR_DEEP"]))
    t.set(Base_Color=col, Roughness=0.42, Specular_IOR_Level=0.40, Coat_Weight=0.10, Coat_Roughness=0.30,
          Sheen_Weight=0.15, Sheen_Roughness=0.4)
    m["game_gloss_scale"] = 0.25
    out["hair"] = m
    return out


# ------------------------------------------------------------------------------------------------ build (bpy)
class _SideAdapter:
    """What `parts.ear` reads of a head loft: t = z, phi 0 = the head's left side at the ear's y."""

    def __init__(self, f, y):
        self.f, self.y = f, y

    def t_at_z(self, z):
        return float(z)

    def pn(self, phi, t):
        t = np.atleast_1d(np.asarray(t, float))
        lo, hi = np.zeros(len(t)), np.full(len(t), 0.12)
        for _ in range(40):
            m = 0.5 * (lo + hi)
            ins = self.f(np.column_stack([m, np.full(len(t), self.y), t])) < 0.0
            lo, hi = np.where(ins, m, lo), np.where(ins, hi, m)
        P = np.column_stack([0.5 * (lo + hi), np.full(len(t), self.y), t])
        return P, np.tile([1.0, 0.0, 0.0], (len(t), 1))


def _mesh(name, V, Q, mat, coll, tris=None, attrs=None, normals=None):
    import kit
    ob = kit.new_mesh(name, V, Q, tris, mat, coll, attrs=attrs)
    if normals is not None:
        ob.data.normals_split_custom_set_from_vertices([tuple(n) for n in unit(normals)])
    return ob


def build_skin(head, mats, prefix, coll, h):
    import time
    t0 = time.time()
    lo, hi = head.box()
    V, Q = mesh_banded(head, lo, hi, h)
    N = head.shading_normals(V)
    attrs = face_masks(head, head.to_head(V))
    ob = _mesh(prefix + "_Head", V, Q, mats["face"], coll, attrs=attrs, normals=N)
    print("anime head: skin %d verts in %.1fs" % (len(V), time.time() - t0))
    return ob


def build_eyes(head, mats, prefix, coll, h):
    """The eye surfaces (her left; mirrored): a grid over the opening in the front view, its depth the uncut skin's
    own surface set back by `inset` (deeper towards the corners: a gently domed eye), closed by a thin back."""
    import kit
    E = head.E
    D = E["dome"]
    ext = D.get("extent", 0.003)   # B2d: how far the grid runs past the opening (it carries the margin under the lids)
    x0, x1 = E["upper"][0][0] - ext, E["lower"][-1][0] + ext
    z0, z1 = min(p[1] for p in E["lower"]) - ext, max(p[1] for p in E["upper"]) + ext
    nu, nv = int(round((x1 - x0) / h)) + 1, int(round((z1 - z0) / h)) + 1
    X, Z = np.meshgrid(np.linspace(x0, x1, nu), np.linspace(z0, z1, nv), indexing="ij")
    Y = head.hit_front(head.base_h, X, Z).reshape(X.shape)
    ec, ea = D["centre"], D["axes"]
    q2 = ((X - ec[0]) / ea[0]) ** 2 + ((Z - ec[1]) / ea[1]) ** 2
    Y = Y + D["inset"] + D["curve"] * q2
    front = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # keep only the opening and a margin round it (beyond, the grid would run off the face's side)
    keep = (head.almond(front) < D.get("margin", 0.0015)).reshape(nu, nv)
    idx = np.arange(nu * nv).reshape(nu, nv)
    qf = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    qk = (keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, 1:] & keep[1:, :-1]).ravel()
    qf = qf[qk]
    used = np.unique(qf)
    remap = np.full(nu * nv, -1)
    remap[used] = np.arange(len(used))
    V, Qa = front[used], remap[qf]
    f = Qa[len(Qa) // 2]
    if np.cross(V[f[1]] - V[f[0]], V[f[3]] - V[f[0]])[1] > 0:
        Qa = Qa[:, ::-1]
    ic, ir = E["iris"]["centre"], E["iris"]["radii"]
    iu = (V[:, 0] - ic[0]) / ir[0]
    iv = (V[:, 2] - ic[1]) / ir[1]
    lid = np.clip((head.U(np.abs(V[:, 0])) - V[:, 2]) / E["iris"]["lid_shadow"], 0.0, 1.0)
    attrs = dict(iu=iu, iv=iv, lid=lid)
    if E.get("paint"):   # B2d: the opening's distance (mm, negative inside) and the position along the eye (0 inner, 1 outer)
        attrs["ec"] = head.almond(V) * 1000.0
        attrs["ex"] = (np.abs(V[:, 0]) - E["upper"][0][0]) / (E["upper"][-1][0] - E["upper"][0][0])
    V = head.world(V)
    ob = _mesh(prefix + "_Eye", V, Qa, mats["eyes"], coll, attrs=attrs)
    kit.solid(ob, D["thick"], bevel=0.0, seg=1)
    ob.modifiers["Solidify"].use_even_offset = False
    kit.mirror(ob)
    return ob


def lash_path(head):
    """The upper lash band's centre line in the front view (head space): x, z, the band's width and the parameter u 0..1
    from the inner end to the flick's tip (shared by the band's geometry and the drawn lash mass, `face_art`)."""
    L = head.E["lash"]
    xs = np.linspace(L["x"][0], L["x"][1], 60)
    zs = head.U(xs)
    fx, fz = L["flick"]
    xf = np.linspace(xs[-1], fx, 12)[1:]
    zf = zs[-1] + (fz - zs[-1]) * ((xf - xs[-1]) / (fx - xs[-1])) ** 1.4
    X, Z = np.concatenate([xs, xf]), np.concatenate([zs, zf])
    u = np.linspace(0.0, 1.0, len(X))
    w = np.interp(u, *zip(*L["width"]))
    return X, Z + w * 0.5 - L["below"], w, u


def build_lashes(head, mats, prefix, coll):
    """The upper lash band: swept along the upper lid on the skin, widening towards the outer corner, ending in a
    flick past it."""
    import kit
    L = head.E["lash"]
    X, Zc, w, u = lash_path(head)
    Y = head.hit_front(head.fh, X, Zc + 0.0004) - L["out"]
    P = np.column_stack([X, Y, Zc])
    N = unit(head.grad(head.base_h, P))
    up = np.tile([0.0, 0.0, 1.0], (len(P), 1))
    up = unit(up - np.sum(up * N, axis=1)[:, None] * N)
    V, Q, T = sweep(P, up, np.maximum(w * 0.5, 0.0001), np.full(len(P), L["thick"] * 0.5), n_u=12)
    ob = _mesh(prefix + "_Lash", head.world(V), Q, mats["lash"], coll, tris=T)
    kit.mirror(ob)
    return ob


class Ear:
    """Her left ear as a field in head space (`spec`: `centre`, `tilt` back and `flare` out in degrees, the auricle as
    two plates (upper, lobe: centre along the ear's (up, back, out) axes, radii), the bowl subtracted from its outer face
    leaving the rim (helix), the deeper concha, and a small tragus in front)."""

    def __init__(self, sp):
        self.sp = sp
        t, f = math.radians(sp["tilt"]), math.radians(sp["flare"])
        up = np.array([0.0, math.sin(t), math.cos(t)])
        out = np.array([math.cos(f), -math.sin(f), 0.0])
        out = unit(out - (out @ up) * up)
        back = np.cross(up, out)
        if back[1] < 0:
            back = -back
        self.A = np.array([up, back, out])
        self.c = np.asarray(sp["centre"], float)

    def local(self, P):
        return (mirror_x(P) - self.c) @ self.A.T

    def __call__(self, P):
        q = self.local(P)
        sp = self.sp
        d = None
        for c, r in sp["plates"]:
            e = ellipsoid(q, c, r)
            d = e if d is None else smin3(d, e, sp["plate_k"])
        for c, r, k in sp["cuts"]:
            d = smax3(d, -ellipsoid(q, c, r), k)
        for c, r, k in sp.get("adds", ()):
            d = smin3(d, ellipsoid(q, c, r), k)
        return d


def build_ears_field(head, spec, mats, prefix, coll):
    """The ears as fields (`Ear`), meshed on their own small grid; her left, mirrored."""
    import kit
    ear = Ear(spec)
    c = np.asarray(spec["centre"], float)
    lo, hi = c - 0.034, c + 0.034
    V, Q = mesh_banded(ear, lo, hi, spec.get("res", 0.0004), block=4)
    N = head.grad(ear, V)
    ob = _mesh(prefix + "_Ear", head.world(V), Q, mats["face"], coll, normals=N)
    kit.mirror(ob)
    return [ob]


def build_ears(head, spec, mats, prefix, coll):
    if "plates" in spec:
        return build_ears_field(head, spec, mats, prefix, coll)
    import parts
    from mathutils import Matrix
    objs = parts.ear(prefix, _SideAdapter(head.base_h, spec["y"]), spec, mats["face"], coll)
    if head.s != 1.0:  # the ear built in head space, placed by its object transform (scaled about the pivot)
        o = head.o
        M = Matrix.Translation(tuple(o)) @ Matrix.Diagonal((head.s, head.s, head.s, 1.0)) @ Matrix.Translation(tuple(-o))
        for ob in objs:
            ob.matrix_world = M
    return objs


def build(prefix, face, eyes, mats, coll="Head", h=0.0007, hair=None, body_objects=None, scale=1.0, hair_face=None):
    """The whole head (`scale`: the head-size variant, see `Head`). Returns the Head (fields) and the objects.
    `hair_face` (B2d): the face table the hair is laid on (its fields: the skin without cuts, the ears), when the face has
    moved on since the hair was fitted -- so the hair stays as built; default the face itself."""
    head = Head(face, eyes, scale)
    objs = [build_skin(head, mats, prefix, coll, h)]
    objs.append(build_eyes(head, mats, prefix, coll, 0.00035))
    objs.append(build_lashes(head, mats, prefix, coll))
    objs += build_ears(head, face["ear"], mats, prefix, coll)
    if hair is not None:
        import animehair
        hh = Head(hair_face, eyes, scale) if hair_face is not None else head
        objs += animehair.build(hh, hair, mats, prefix, coll, body_objects)
    return head, objs
